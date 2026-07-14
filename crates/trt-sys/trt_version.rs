use std::collections::HashMap;

/// Parse "MAJOR.MINOR.PATCH.BUILD" from NvInferVersion.h so the version
/// constant tracks the actually-installed TRT (engine-cache keys depend on it).
#[allow(dead_code)] // Used by build.rs; the lib.rs test include exercises the pure text parser.
pub(crate) fn parse_trt_version(trt_inc: &str) -> Option<String> {
    let text = std::fs::read_to_string(format!("{trt_inc}/NvInferVersion.h")).ok()?;
    parse_trt_version_text(&text)
}

pub(crate) fn parse_trt_version_text(text: &str) -> Option<String> {
    let defines: HashMap<&str, &str> = text
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next()? != "#define" {
                return None;
            }
            Some((words.next()?, words.next()?))
        })
        .collect();
    let grab = |name: &str| -> Option<u32> {
        let mut value = *defines.get(name)?;
        for _ in 0..16 {
            if let Ok(number) = value.parse() {
                return Some(number);
            }
            value = *defines.get(value)?;
        }
        None
    };
    Some(format!(
        "{}.{}.{}.{}",
        grab("NV_TENSORRT_MAJOR")?,
        grab("NV_TENSORRT_MINOR")?,
        grab("NV_TENSORRT_PATCH")?,
        grab("NV_TENSORRT_BUILD")?,
    ))
}
