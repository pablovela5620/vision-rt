use rerun::{
    Boxes2D, ChannelDatatype, DepthImage, EncodedImage, MediaType, Pinhole, Points3D,
    RecordingStream, RecordingStreamBuilder, RecordingStreamError, ViewCoordinates,
};
use vrt_track::{CameraIntrinsics, Track, TrackState};
use vrt_types::DepthImage as VrtDepthImage;
use vrt_viz::{track_color, JpegEncoder, VizError};

#[derive(Debug, thiserror::Error)]
pub enum RerunLogError {
    #[error(transparent)]
    Recording(#[from] RecordingStreamError),
    #[error(transparent)]
    Encode(#[from] VizError),
    #[error("frame index exceeds Rerun's signed sequence range")]
    FrameIndex(#[from] std::num::TryFromIntError),
    #[error(
        "depth grid changed after its static calibration was logged: expected {expected:?}, got {actual:?}"
    )]
    DepthGridChange {
        expected: (usize, usize),
        actual: (usize, usize),
    },
}

pub struct RerunLog {
    rec: RecordingStream,
    jpeg: JpegEncoder,
    intr: CameraIntrinsics,
    width: usize,
    height: usize,
    depth_grid: Option<(usize, usize)>,
}

impl RerunLog {
    pub fn from_output(
        output: &str,
        intr: &CameraIntrinsics,
        width: usize,
        height: usize,
    ) -> Result<Self, RerunLogError> {
        let builder = RecordingStreamBuilder::new("vrt_track");
        let rec = if output.ends_with(".rrd") {
            builder.save(output)?
        } else if output == "rerun" {
            builder.spawn()?
        } else {
            builder.connect_grpc_opts(output)?
        };
        let pinhole = Pinhole::new([
            [intr.fx, 0.0, 0.0],
            [0.0, intr.fy, 0.0],
            [intr.cx, intr.cy, 1.0],
        ])
        .with_resolution([width as f32, height as f32]);
        rec.log_static("world/camera", &pinhole)?;
        rec.log_static("world/camera", &ViewCoordinates::RDF())?;
        Ok(Self {
            rec,
            jpeg: JpegEncoder::new(85)?,
            intr: *intr,
            width,
            height,
            depth_grid: None,
        })
    }

    pub fn log_frame(
        &mut self,
        frame: u64,
        rgb: Vec<u8>,
        depth: &VrtDepthImage,
        tracks: &[Track],
    ) -> Result<(), RerunLogError> {
        self.rec.set_time_sequence("frame", i64::try_from(frame)?);
        let jpeg = self.jpeg.encode(rgb, self.width, self.height)?;
        self.rec.log(
            "world/camera/image",
            &EncodedImage::new(jpeg).with_media_type(MediaType::jpeg()),
        )?;

        let confirmed: Vec<&Track> = tracks
            .iter()
            .filter(|track| track.state == TrackState::Confirmed)
            .collect();
        let mins: Vec<[f32; 2]> = confirmed
            .iter()
            .map(|track| [track.bbox[0], track.bbox[1]])
            .collect();
        let sizes: Vec<[f32; 2]> = confirmed
            .iter()
            .map(|track| [track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1]])
            .collect();
        let colors: Vec<[u8; 3]> = confirmed
            .iter()
            .map(|track| track_color(track.id))
            .collect();
        let labels: Vec<String> = confirmed
            .iter()
            .map(|track| format!("#{} {}", track.id, super::coco_name(track.class_id)))
            .collect();
        self.rec.log(
            "world/camera/image/tracks",
            &Boxes2D::from_mins_and_sizes(mins, sizes)
                .with_colors(colors.iter().copied())
                .with_labels(labels.iter().map(String::as_str)),
        )?;

        let size = depth.size();
        let grid = (size.width, size.height);
        match self.depth_grid {
            None => {
                // Both models Stretch the full frame, so depth-grid calibration is
                // the camera pinhole scaled by the plain per-axis grid/frame ratios.
                // Sibling of world/camera, NOT a child: rerun cannot chain a pinhole
                // under another pinhole ("no transform path to the view's target
                // frame"), and both cameras sit at the world origin anyway.
                let sx = size.width as f32 / self.width as f32;
                let sy = size.height as f32 / self.height as f32;
                let depth_pinhole = Pinhole::new([
                    [self.intr.fx * sx, 0.0, 0.0],
                    [0.0, self.intr.fy * sy, 0.0],
                    [self.intr.cx * sx, self.intr.cy * sy, 1.0],
                ])
                .with_resolution([size.width as f32, size.height as f32]);
                self.rec.log_static("world/depth", &depth_pinhole)?;
                self.rec.log_static("world/depth", &ViewCoordinates::RDF())?;
                self.depth_grid = Some(grid);
            }
            Some(expected) if expected != grid => {
                return Err(RerunLogError::DepthGridChange {
                    expected,
                    actual: grid,
                });
            }
            Some(_) => {}
        }
        let depth_bytes = bytemuck::cast_slice(depth.as_slice()).to_vec();
        self.rec.log(
            "world/depth",
            &DepthImage::from_data_type_and_bytes(
                depth_bytes,
                [size.width as u32, size.height as u32],
                ChannelDatatype::F32,
            )
            .with_meter(1.0),
        )?;

        let positions: Vec<[f32; 3]> = confirmed
            .iter()
            .map(|track| track.metric_position(&self.intr))
            .collect();
        self.rec.log(
            "world/tracks",
            &Points3D::new(positions)
                .with_colors(colors)
                .with_labels(labels.iter().map(String::as_str)),
        )?;
        Ok(())
    }
}
