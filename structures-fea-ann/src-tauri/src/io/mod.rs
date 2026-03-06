use std::fs;
use std::path::{Path, PathBuf};

use crate::contracts::{ExportResult, ReportInput};

pub fn export_report(input: &ReportInput) -> Result<ExportResult, String> {
    let format = input.format.to_lowercase();
    let target = normalize_path(&input.path);

    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create output directory: {e}"))?;
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
            csv.push_str(&format!("length_in,{}\n", input.solve_input.geometry.length_in));
            csv.push_str(&format!("width_in,{}\n", input.solve_input.geometry.width_in));
            csv.push_str(&format!("thickness_in,{}\n", input.solve_input.geometry.thickness_in));
            csv.push_str(&format!("axial_load_lbf,{}\n", input.solve_input.load.axial_load_lbf));
            if let Some(fem) = &input.fem_result {
                csv.push_str(&format!("fem_von_mises_psi,{}\n", fem.von_mises_psi));
                csv.push_str(&format!("fem_max_principal_psi,{}\n", fem.max_principal_psi));
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
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")).join(p)
    }
}

fn minimal_pdf_text(input: &ReportInput) -> String {
    let mut lines = vec![
        "Structures FEA + ANN Report".to_string(),
        format!("Length (in): {:.4}", input.solve_input.geometry.length_in),
        format!("Width (in): {:.4}", input.solve_input.geometry.width_in),
        format!("Thickness (in): {:.4}", input.solve_input.geometry.thickness_in),
        format!("Axial Load (lbf): {:.4}", input.solve_input.load.axial_load_lbf),
    ];

    if let Some(fem) = &input.fem_result {
        lines.push(format!("FEM Von Mises (psi): {:.4}", fem.von_mises_psi));
        lines.push(format!("FEM Max Principal (psi): {:.4}", fem.max_principal_psi));
    }

    if let Some(ann) = &input.ann_result {
        lines.push(format!("ANN Confidence: {:.4}", ann.confidence));
        lines.push(format!("ANN Uncertainty: {:.4}", ann.uncertainty));
    }

    lines.join("\\n")
}

fn wrap_as_basic_pdf(text: &str) -> Vec<u8> {
    let escaped = text.replace("(", "\\(").replace(")", "\\)");
    let content = format!("BT /F1 12 Tf 50 760 Td ({}) Tj ET", escaped.replace('\n', ") Tj T* ("));

    let mut pdf = Vec::new();
    let mut offsets = Vec::new();

    let write_obj = |pdf: &mut Vec<u8>, offsets: &mut Vec<usize>, obj_num: usize, body: &str| {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", obj_num, body).as_bytes());
    };

    pdf.extend_from_slice(b"%PDF-1.4\n");
    write_obj(&mut pdf, &mut offsets, 1, "<< /Type /Catalog /Pages 2 0 R >>");
    write_obj(&mut pdf, &mut offsets, 2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    write_obj(
        &mut pdf,
        &mut offsets,
        3,
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    );
    write_obj(&mut pdf, &mut offsets, 4, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>");
    write_obj(
        &mut pdf,
        &mut offsets,
        5,
        &format!("<< /Length {} >>\nstream\n{}\nendstream", content.len(), content),
    );

    let xref_start = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1).as_bytes());
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
