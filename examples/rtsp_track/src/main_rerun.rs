//! Video file → RF-DETR-Seg + Depth Anything V2 + **3D tracking** → Rerun.
//!
//! The seg + depth pipeline is the one-stream / one-sync flow of `rtsp_depth`; this
//! example adds the tracker on top. The mask-sampled metric depth feeds each
//! detection's `pz` (`Detection::with_depth`), so the 3D Kalman carries a real
//! metric-depth axis + approach velocity per track. Each frame logs the rectified
//! image, tracked boxes, metric depth, and 3D track positions to Rerun.
//!
//! ```text
//!   source → seg.submit + depth.submit → sample_masks → ONE sync
//!   readout → Detection::with_depth(z) → Tracker::update → [Track]
//!   Rerun image + Boxes2D + DepthImage + Points3D
//! ```
//!
//! Build:  export CARGO_NET_GIT_FETCH_WITH_CLI=true
//!   cargo run --release --manifest-path examples/rtsp_track/Cargo.toml \
//!       --bin track_rerun -- <seg.engine|seg.onnx|hub> \
//!       <depth.engine|depth.onnx|hub> <video-file> [conf] [out]
//!
//! The optional output is `rerun` (spawn a viewer), `rerun+http://…` (connect),
//! or an `*.rrd` path (save). If omitted, a viewer is spawned.

mod file_source;
mod rerun_log;

use std::collections::HashSet;
use std::time::Instant;

use cudarc::driver::CudaContext;
use file_source::FileSource;
use kornia_image::{Image, ImageSize};
use vrt_depth_anything::DepthAnything;
use vrt_rfdetr_seg::RfDetrSeg;
use vrt_track::{CameraIntrinsics, Detection, TrackState, Tracker, TrackerConfig};
use vrt_types::Undistorter;

type Res<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Generic horizontal-FoV default for arbitrary video files (Tapo C210 ≈ 67°).
const DEFAULT_HFOV_DEG: f32 = 67.0;
fn main() -> Res<()> {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: track_rerun <seg> <depth> <video-file> [conf] [out]");
        std::process::exit(1);
    }
    let (seg_engine, depth_engine, source_arg) = (&args[1], &args[2], &args[3]);
    let conf: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.4);
    let output = args.get(5).map_or("rerun", String::as_str);

    // One shared CUDA stream: source copy, seg, depth, and the depth-at-mask fusion
    // all enqueue on it, so a single sync completes the frame's GPU work.
    let stream = CudaContext::new(0)?.default_stream();
    let mut source = FileSource::open(source_arg, stream.clone())?;
    let (w, h) = (source.width() as usize, source.height() as usize);
    println!("stream {w}x{h} → RF-DETR-Seg (conf ≥ {conf}) + Depth Anything V2 + 3D tracker");

    let mut seg = if seg_engine == "hub" {
        RfDetrSeg::from_hub(stream.clone(), conf)?
    } else if seg_engine.ends_with(".onnx") {
        RfDetrSeg::from_onnx(seg_engine, stream.clone(), conf)?
    } else {
        RfDetrSeg::from_engine_file(seg_engine, stream.clone(), conf)?
    };
    let mut depth = if depth_engine == "hub" {
        DepthAnything::from_hub(stream.clone())?
    } else if depth_engine.ends_with(".onnx") {
        DepthAnything::from_onnx(depth_engine, stream.clone())?
    } else {
        DepthAnything::from_engine_file(depth_engine, stream.clone())?
    };
    let mut d = seg.alloc_result()?;
    let mut z = depth.alloc_result()?;
    // Diagnostic A/B: `RTSP_TRACK_NO_DEPTH=1` disables the depth gate + soft cost so
    // ID stability can be compared with vs without the 3D association terms.
    let mut cfg = TrackerConfig::default();
    if std::env::var("RTSP_TRACK_NO_DEPTH").is_ok() {
        cfg.depth_gate = false; // A/B toggle for the depth-gate's effect on ID stability
        println!("(depth gate DISABLED for A/B)");
    }
    let mut tracker = Tracker::new(cfg)?;
    let env_f32 = |name: &str, default: f32| {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    };
    let hfov_deg = env_f32("VRT_TRACK_HFOV_DEG", DEFAULT_HFOV_DEG);
    let k1 = env_f32("VRT_TRACK_K1", 0.0);
    let intr = CameraIntrinsics::from_hfov(w as f32, h as f32, hfov_deg);
    // Lens undistort → rectified pinhole; identity unless VRT_TRACK_K1 is set.
    let undist = Undistorter::new(&intr, k1, w, h, &stream)?;
    let mut rect = Image::<u8, 3>::zeros_cuda(
        ImageSize {
            width: w,
            height: h,
        },
        &stream,
    )?; // reused

    if !is_rerun_output(output) {
        return Err(format!("invalid Rerun output: {output}").into());
    }
    let mut rerun_sink = rerun_log::RerunLog::from_output(output, &intr, w, h)?;

    let mut dets: Vec<Detection> = Vec::new();
    let mut window_ids: HashSet<u64> = HashSet::new(); // distinct ids per 100-frame window
    let ms = |dur: std::time::Duration| dur.as_secs_f64() * 1e3;
    let (mut n, t_start) = (0u64, Instant::now());
    let (mut a_src, mut a_enq, mut a_fus, mut a_sync, mut a_read, mut a_trk) =
        (0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut prev, mut ema_dt, mut a_dt) = (Instant::now(), 0.0f64, 0.0f64);
    let mut a_log = 0.0f64;
    loop {
        let t0 = Instant::now();
        let Some(frame) = source.next_frame()? else {
            break;
        };
        let t1 = Instant::now();
        undist.apply(frame.image(), &mut rect, &stream)?; // rectify on the shared stream
        seg.submit(&rect, &mut d)?;
        depth.submit(&rect, &mut z)?;
        let t2 = Instant::now();
        let zs = z.depth_image().sample_masks(
            d.masks_slice(),
            d.mask_size(),
            d.count_slice(),
            &stream,
        )?;
        let t3 = Instant::now();
        stream.synchronize()?; // the one sync completes source + detect + depth + fusion
        let t4 = Instant::now();

        // Readout: survivor boxes + per-instance metric depth (live-count prefix only).
        let inst_n = d.count();
        let detections = d.detections()?;
        let z_m = stream.clone_dtoh(&zs.slice(0..inst_n))?;
        let t5 = Instant::now();

        // Feed the tracker: each box carries its mask-sampled metric depth → `pz`.
        dets.clear();
        for (det, &zv) in detections.iter().zip(&z_m) {
            let mut det = Detection::new(det.bbox, det.score, det.class_id);
            if zv > 0.0 {
                det = det.with_depth(zv);
            }
            dets.push(det);
        }
        // Real inter-frame dt in nominal-frame units (EMA-calibrated) → jitter-robust.
        let interval = t0.duration_since(prev).as_secs_f64();
        prev = t0;
        let dt = if ema_dt > 0.0 {
            (interval / ema_dt).clamp(0.25, 4.0)
        } else {
            1.0
        };
        if interval > 1e-3 {
            ema_dt = if ema_dt > 0.0 {
                0.9 * ema_dt + 0.1 * interval
            } else {
                interval
            };
        }
        let tracks = tracker.update_dt(&dets, dt);
        let t6 = Instant::now();
        for t in tracks.iter().filter(|t| t.state == TrackState::Confirmed) {
            window_ids.insert(t.id); // churn: distinct ids over the window vs live count
        }

        let host = rect.to_host(&stream)?.into_vec();
        let depth_host = z.depth_host()?;
        rerun_sink.log_frame(n, host, &depth_host, &tracks)?;
        a_log += ms(Instant::now() - t6);

        n += 1;
        a_src += ms(t1 - t0);
        a_enq += ms(t2 - t1);
        a_fus += ms(t3 - t2);
        a_sync += ms(t4 - t3);
        a_read += ms(t5 - t4);
        a_trk += ms(t6 - t5);
        a_dt += dt;
        if n.is_multiple_of(100) {
            // Per-window profiling + track/churn stats — silent by default, opt in with
            // `RUST_LOG=track_rerun=debug`.
            if log::log_enabled!(log::Level::Debug) {
                let k = 100.0;
                let confirmed = tracks
                    .iter()
                    .filter(|t| t.state == TrackState::Confirmed)
                    .count();
                // Churn: distinct confirmed ids this window vs the live count. Static
                // scene ⇒ distinct ≈ live; distinct ≫ live ⇒ ids are switching.
                let distinct = window_ids.len();
                log::debug!(
                    "{n} frames | {:.1} fps | source {:.2} | enqueue {:.3} | fusion {:.3} | \
                     sync(GPU) {:.2} | readout {:.3} | track {:.3} | dt {:.2} | {inst_n} det → \
                     {confirmed} conf | {distinct} distinct-ids/100f",
                    n as f64 / t_start.elapsed().as_secs_f64(),
                    a_src / k,
                    a_enq / k,
                    a_fus / k,
                    a_sync / k,
                    a_read / k,
                    a_trk / k,
                    a_dt / k,
                );
                let spf = if ema_dt > 0.0 {
                    ema_dt as f32
                } else {
                    1.0 / 15.0
                };
                let shown: Vec<String> = tracks
                    .iter()
                    .filter(|t| t.state == TrackState::Confirmed)
                    .take(6)
                    .map(|t| {
                        let [x, y, zz] = t.metric_position(&intr);
                        let mv = t.metric_velocity(&intr);
                        let speed = (mv[0].powi(2) + mv[1].powi(2) + mv[2].powi(2)).sqrt() / spf;
                        format!(
                            "#{} {} [X{x:+.1} Y{y:+.1} Z{zz:.1}]m {speed:.1}m/s",
                            t.id,
                            coco_name(t.class_id)
                        )
                    })
                    .collect();
                log::debug!("tracks: {}", shown.join(", "));
                log::debug!("rerun log {:.1} ms", a_log / k);
            }
            window_ids.clear();
            (a_src, a_enq, a_fus, a_sync, a_read, a_trk, a_dt, a_log) =
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
    }
    Ok(())
}

fn is_rerun_output(s: &str) -> bool {
    s.ends_with(".rrd") || s == "rerun" || s.starts_with("rerun+http")
}

/// COCO 91-class category-id → name (for the stdout track list).
fn coco_name(id: u32) -> &'static str {
    COCO91.get(id as usize).copied().unwrap_or("?")
}

const COCO91: [&str; 91] = [
    "background",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];
