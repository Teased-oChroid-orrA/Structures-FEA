use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::contracts::{
    CheckpointRetentionPolicy, ExportResult, ModelStatus, PurgeCheckpointsResult, ReportInput,
    TrainingCheckpointInfo,
};
use crate::pinn::UniversalPinnState;

pub fn export_report(input: &ReportInput) -> Result<ExportResult, String> {
    let format = input.format.to_lowercase();
    let target = normalize_path(&input.path);

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory: {e}"))?;
    }

    let bytes = match format.as_str() {
        "json" => {
            let payload = serde_json::to_vec_pretty(input)
                .map_err(|e| format!("JSON serialize failed: {e}"))?;
            fs::write(&target, &payload).map_err(|e| format!("Write failed: {e}"))?;
            payload.len()
        }
        "csv" => {
            let mut csv = String::new();
            csv.push_str("field,value\n");
            csv.push_str(&format!(
                "length_in,{}\n",
                input.solve_input.geometry.length_in
            ));
            csv.push_str(&format!(
                "width_in,{}\n",
                input.solve_input.geometry.width_in
            ));
            csv.push_str(&format!(
                "thickness_in,{}\n",
                input.solve_input.geometry.thickness_in
            ));
            csv.push_str(&format!(
                "axial_load_lbf,{}\n",
                input.solve_input.load.axial_load_lbf
            ));
            if let Some(fem) = &input.fem_result {
                csv.push_str(&format!("fem_von_mises_psi,{}\n", fem.von_mises_psi));
                csv.push_str(&format!(
                    "fem_max_principal_psi,{}\n",
                    fem.max_principal_psi
                ));
            }
            if let Some(ann) = &input.ann_result {
                csv.push_str(&format!("ann_confidence,{}\n", ann.confidence));
                csv.push_str(&format!("ann_uncertainty,{}\n", ann.uncertainty));
            }
            fs::write(&target, &csv).map_err(|e| format!("Write failed: {e}"))?;
            csv.len()
        }
        "pdf" => {
            let text = minimal_pdf_text(input);
            let payload = wrap_as_basic_pdf(&text);
            fs::write(&target, &payload).map_err(|e| format!("Write failed: {e}"))?;
            payload.len()
        }
        _ => return Err("Unsupported format. Use json, csv, or pdf.".to_string()),
    };

    Ok(ExportResult {
        path: target.display().to_string(),
        bytes_written: bytes,
        format,
    })
}

fn normalize_path(raw: &str) -> PathBuf {
    let p = Path::new(raw);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(p)
    }
}

fn minimal_pdf_text(input: &ReportInput) -> String {
    let mut lines = vec![
        "Structures FEA + ANN Report".to_string(),
        format!("Length (in): {:.4}", input.solve_input.geometry.length_in),
        format!("Width (in): {:.4}", input.solve_input.geometry.width_in),
        format!(
            "Thickness (in): {:.4}",
            input.solve_input.geometry.thickness_in
        ),
        format!(
            "Axial Load (lbf): {:.4}",
            input.solve_input.load.axial_load_lbf
        ),
    ];

    if let Some(fem) = &input.fem_result {
        lines.push(format!("FEM Von Mises (psi): {:.4}", fem.von_mises_psi));
        lines.push(format!(
            "FEM Max Principal (psi): {:.4}",
            fem.max_principal_psi
        ));
    }

    if let Some(ann) = &input.ann_result {
        lines.push(format!("ANN Confidence: {:.4}", ann.confidence));
        lines.push(format!("ANN Uncertainty: {:.4}", ann.uncertainty));
    }

    lines.join("\\n")
}

fn wrap_as_basic_pdf(text: &str) -> Vec<u8> {
    let escaped = text.replace("(", "\\(").replace(")", "\\)");
    let content = format!(
        "BT /F1 12 Tf 50 760 Td ({}) Tj ET",
        escaped.replace('\n', ") Tj T* (")
    );

    let mut pdf = Vec::new();
    let mut offsets = Vec::new();

    let write_obj = |pdf: &mut Vec<u8>, offsets: &mut Vec<usize>, obj_num: usize, body: &str| {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", obj_num, body).as_bytes());
    };

    pdf.extend_from_slice(b"%PDF-1.4\n");
    write_obj(
        &mut pdf,
        &mut offsets,
        1,
        "<< /Type /Catalog /Pages 2 0 R >>",
    );
    write_obj(
        &mut pdf,
        &mut offsets,
        2,
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    );
    write_obj(
        &mut pdf,
        &mut offsets,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    );
    write_obj(
        &mut pdf,
        &mut offsets,
        4,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    );
    write_obj(
        &mut pdf,
        &mut offsets,
        5,
        &format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len(),
            content
        ),
    );

    let xref_start = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes(),
    );
    for off in offsets {
        pdf.extend_from_slice(format!("{:010} 00000 n \n", off).as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            6, xref_start
        )
        .as_bytes(),
    );

    pdf
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct CheckpointPayload {
    info: TrainingCheckpointInfo,
    model: UniversalPinnState,
    model_status: ModelStatus,
}

fn parse_checkpoint_payload(bytes: &[u8]) -> Result<CheckpointPayload, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("Checkpoint parse failed: {e}"))
}

fn read_checkpoint_payload(path: &Path) -> Result<CheckpointPayload, String> {
    let bytes = fs::read(path)
        .map_err(|e| format!("Failed to read checkpoint file {}: {e}", path.display()))?;
    parse_checkpoint_payload(&bytes)
        .map_err(|e| format!("{} ({})", e, path.display()))
}

fn checkpoints_dir() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("outputs")
        .join("checkpoints")
}

pub fn save_training_checkpoint(
    tag: Option<String>,
    model_state: &UniversalPinnState,
    model_status: &ModelStatus,
    epoch: usize,
    mark_best: bool,
) -> Result<TrainingCheckpointInfo, String> {
    let dir = checkpoints_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("Failed to create checkpoints directory: {e}"))?;
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Clock error: {e}"))?
        .as_millis();
    let safe_tag = tag.unwrap_or_else(|| "manual".to_string());
    let id = format!("cp-{}-{}", now_ms, model_status.model_version);
    let file_name = format!("{}-{}.json", id, safe_tag.replace(' ', "_"));
    let path = dir.join(file_name);
    let info = TrainingCheckpointInfo {
        id: id.clone(),
        tag: safe_tag,
        path: path.display().to_string(),
        created_epoch: epoch,
        model_version: model_status.model_version,
        best_val_loss: model_state
            .burn
            .as_ref()
            .map(|burn| burn.best_val_loss)
            .unwrap_or(model_state.ann.best_val_loss),
        is_best: mark_best,
        created_at_unix_ms: now_ms,
    };
    let payload = CheckpointPayload {
        info: info.clone(),
        model: model_state.clone(),
        model_status: model_status.clone(),
    };
    let data = serde_json::to_vec_pretty(&payload)
        .map_err(|e| format!("Failed to serialize checkpoint payload: {e}"))?;
    fs::write(&path, data).map_err(|e| format!("Failed to write checkpoint: {e}"))?;
    Ok(info)
}

pub fn list_training_checkpoints() -> Result<Vec<TrainingCheckpointInfo>, String> {
    let dir = checkpoints_dir();
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut items = Vec::new();
    let rd =
        fs::read_dir(&dir).map_err(|e| format!("Failed to read checkpoints directory: {e}"))?;
    for entry in rd {
        let entry = entry.map_err(|e| format!("Failed to read checkpoint entry: {e}"))?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let payload = read_checkpoint_payload(&p)?;
        items.push(payload.info);
    }
    items.sort_by(|a, b| b.created_at_unix_ms.cmp(&a.created_at_unix_ms));
    Ok(items)
}

pub fn load_training_checkpoint(
    id: &str,
) -> Result<(TrainingCheckpointInfo, UniversalPinnState, ModelStatus), String> {
    let dir = checkpoints_dir();
    if !dir.exists() {
        return Err("No checkpoints directory found".to_string());
    }
    let rd =
        fs::read_dir(&dir).map_err(|e| format!("Failed to read checkpoints directory: {e}"))?;
    for entry in rd {
        let entry = entry.map_err(|e| format!("Failed to read checkpoint entry: {e}"))?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let payload = read_checkpoint_payload(&p)?;
        if payload.info.id == id {
            return Ok((payload.info, payload.model, payload.model_status));
        }
    }
    Err(format!("Checkpoint not found: {id}"))
}

pub fn purge_training_checkpoints(
    retention_policy: &CheckpointRetentionPolicy,
) -> Result<PurgeCheckpointsResult, String> {
    let dir = checkpoints_dir();
    if !dir.exists() {
        return Ok(PurgeCheckpointsResult {
            removed: 0,
            kept: 0,
        });
    }
    let keep_last = retention_policy.keep_last.max(1);
    let keep_best = retention_policy.keep_best;
    let mut all = list_training_checkpoints()?;
    let total = all.len();
    let mut kept = Vec::new();
    for cp in all.drain(..).take(keep_last) {
        kept.push(cp.id);
    }
    if keep_best > 0 {
        let mut best = list_training_checkpoints()?;
        best.sort_by(|a, b| {
            a.best_val_loss
                .partial_cmp(&b.best_val_loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for cp in best.into_iter().take(keep_best) {
            if !kept.iter().any(|k| k == &cp.id) {
                kept.push(cp.id);
            }
        }
    }
    let rd =
        fs::read_dir(&dir).map_err(|e| format!("Failed to read checkpoints directory: {e}"))?;
    let mut removed = 0usize;
    for entry in rd {
        let entry = entry.map_err(|e| format!("Failed to read checkpoint entry: {e}"))?;
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let payload = read_checkpoint_payload(&p)?;
        if !kept.iter().any(|k| k == &payload.info.id) {
            fs::remove_file(&p).map_err(|e| format!("Failed to remove checkpoint file: {e}"))?;
            removed += 1;
        }
    }
    Ok(PurgeCheckpointsResult {
        removed,
        kept: total.saturating_sub(removed),
    })
}

pub fn resume_training_from_checkpoint(
    id: &str,
) -> Result<(TrainingCheckpointInfo, UniversalPinnState, ModelStatus), String> {
    load_training_checkpoint(id)
}
