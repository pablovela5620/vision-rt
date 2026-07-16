//! GStreamer/kornia-io file decoding at native resolution with paced playback.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaStream, DriverError, PinnedHostSlice};
use kornia_image::{Image, ImageError, ImageSize};
use kornia_io::gstreamer::error::VideoReaderError;
use kornia_io::gstreamer::video::{ImageFormat, VideoReader};

const FRAME_POLL_INTERVAL: Duration = Duration::from_millis(1);
const MIN_EOS_STALL: Duration = Duration::from_millis(500);
const PLAYING_STALL_CAP: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum FileSourceError {
    #[error("failed to open video {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: FileOpenError,
    },
    #[error("video frame decode failed")]
    Decode(#[from] VideoReaderError),
    #[error("pinned host buffer failed")]
    Pinned(#[source] DriverError),
    #[error("CUDA frame upload failed")]
    Upload {
        #[source]
        source: Option<DriverError>,
    },
    #[error("video resolution changed: expected {expected} bytes, got {actual}")]
    ResolutionChange { expected: usize, actual: usize },
    #[error("video stopped yielding frames before clean EOS could be confirmed")]
    Stall,
}

#[derive(Debug, thiserror::Error)]
pub enum FileOpenError {
    #[error("file does not exist")]
    Missing,
    #[error("decoder setup failed")]
    Decoder(#[source] VideoReaderError),
    #[error("video reached EOS before its first frame")]
    Empty,
    #[error("video dimensions do not fit the platform")]
    Dimensions(#[source] std::num::TryFromIntError),
    #[error("video frame dimensions overflow")]
    DimensionOverflow,
    #[error("device image allocation failed")]
    DeviceImage(#[source] ImageError),
}

/// Kornia-decoded RGB frames uploaded into one reusable device image. A returned
/// frame borrows that image, keeping it alive through the caller's shared-stream sync.
pub struct FileSource {
    width: u32,
    height: u32,
    stream: Arc<CudaStream>,
    reader: VideoReader,
    first_frame: Option<Image<u8, 3>>,
    progress: PlaybackProgress,
    host: PinnedHostSlice<u8>,
    image: Image<u8, 3>,
}

impl FileSource {
    pub fn open(path: &str, stream: Arc<CudaStream>) -> Result<Self, FileSourceError> {
        let open_error = |source| FileSourceError::Open {
            path: path.to_owned(),
            source,
        };
        if !Path::new(path).is_file() {
            return Err(open_error(FileOpenError::Missing));
        }

        let mut reader = VideoReader::new(path, ImageFormat::Rgb8)
            .map_err(|source| open_error(FileOpenError::Decoder(source)))?;
        reader
            .start()
            .map_err(|source| open_error(FileOpenError::Decoder(source)))?;
        let mut progress = PlaybackProgress::new();
        let first_frame = next_decoded(&mut reader, &mut progress)?
            .ok_or_else(|| open_error(FileOpenError::Empty))?;

        let size = first_frame.size();
        let width = u32::try_from(size.width)
            .map_err(|source| open_error(FileOpenError::Dimensions(source)))?;
        let height = u32::try_from(size.height)
            .map_err(|source| open_error(FileOpenError::Dimensions(source)))?;
        let len = size
            .width
            .checked_mul(size.height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| open_error(FileOpenError::DimensionOverflow))?;
        // SAFETY: every byte is initialized from a decoded frame before upload.
        let host =
            unsafe { stream.context().alloc_pinned::<u8>(len) }.map_err(FileSourceError::Pinned)?;
        let image = Image::zeros_cuda(
            ImageSize {
                width: size.width,
                height: size.height,
            },
            &stream,
        )
        .map_err(|source| open_error(FileOpenError::DeviceImage(source)))?;

        Ok(Self {
            width,
            height,
            stream,
            reader,
            first_frame: Some(first_frame),
            progress,
            host,
            image,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn next_frame(&mut self) -> Result<Option<FileFrame<'_>>, FileSourceError> {
        let frame = match self.first_frame.take() {
            Some(frame) => frame,
            None => match next_decoded(&mut self.reader, &mut self.progress)? {
                Some(frame) => frame,
                None => return Ok(None),
            },
        };

        let decoded = frame.as_slice();
        let host = self.host.as_mut_slice().map_err(FileSourceError::Pinned)?;
        if decoded.len() != host.len() {
            return Err(FileSourceError::ResolutionChange {
                expected: host.len(),
                actual: decoded.len(),
            });
        }
        host.copy_from_slice(decoded);

        let device = self
            .image
            .as_cudaslice_mut()
            .ok_or(FileSourceError::Upload { source: None })?;
        self.stream
            .memcpy_htod(&self.host, device)
            .map_err(|source| FileSourceError::Upload {
                source: Some(source),
            })?;
        Ok(Some(FileFrame { image: &self.image }))
    }
}

struct PlaybackProgress {
    last_position: Option<Duration>,
    last_advance: Instant,
    playing_without_frame: Option<Instant>,
}

impl PlaybackProgress {
    fn new() -> Self {
        Self {
            last_position: None,
            last_advance: Instant::now(),
            playing_without_frame: None,
        }
    }

    fn note_frame(&mut self, reader: &VideoReader) {
        self.playing_without_frame = None;
        if let Some(position) = reader.get_pos() {
            self.last_position = Some(position);
            self.last_advance = Instant::now();
        }
    }
}

fn next_decoded(
    reader: &mut VideoReader,
    progress: &mut PlaybackProgress,
) -> Result<Option<Image<u8, 3>>, FileSourceError> {
    loop {
        if let Some(frame) = reader.grab_rgb8()? {
            progress.note_frame(reader);
            return Ok(Some(frame));
        }
        // Non-Playing is NOT an instant failure: set_state(Playing) is async, so the
        // pipeline reports Ready/Paused while prerolling (notably on the very first
        // poll). Keep waiting — the stall cap below bounds a pipeline that never
        // reaches Playing, and the position check handles EOS in any state.

        let now = Instant::now();
        let empty_since = *progress.playing_without_frame.get_or_insert(now);
        if now.duration_since(empty_since) >= PLAYING_STALL_CAP {
            return Err(FileSourceError::Stall);
        }

        // StreamCapture never watches the GStreamer bus, so EOS leaves the pipeline
        // Playing. A stable position for max(3 frame intervals, 500ms) is clean EOS:
        // the window also lets the appsink callback publish a queued final sample.
        if let Some(position) = reader.get_pos() {
            if progress
                .last_position
                .is_none_or(|previous| position > previous)
            {
                progress.last_position = Some(position);
                progress.last_advance = now;
            } else {
                let fps = reader.get_fps().filter(|fps| fps.is_finite() && *fps > 0.0);
                let frame_window = fps
                    .map(|fps| Duration::from_secs_f64(3.0 / fps))
                    .unwrap_or(MIN_EOS_STALL);
                if now.duration_since(progress.last_advance) >= frame_window.max(MIN_EOS_STALL) {
                    return Ok(None);
                }
            }
        }
        std::thread::sleep(FRAME_POLL_INTERVAL);
    }
}

/// A device-resident file frame valid until its borrow ends after stream sync.
pub struct FileFrame<'a> {
    image: &'a Image<u8, 3>,
}

impl FileFrame<'_> {
    pub fn image(&self) -> &Image<u8, 3> {
        self.image
    }
}
