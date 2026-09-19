export const MOCK_UPLOAD_RESPONSE = {
  status: "success",
  tile_id: "sample_forest_tile",
  shape: [12, 120, 120],
  modality: "MULTIMODAL_S1_S2",
  detection_basis: "12 channels detected",
  basis_label: "12-Channel Sentinel-2 + Sentinel-1 SAR",
  display_label: "12-channel S2+S1",
  available_analyses: ["spectral", "classification", "sar", "vlm"],
};

export const MOCK_CHAT_RESPONSE = {
  tile_id: "sample_forest_tile",
  query: "What is visible in this image?",
  response: "Here's the analysis of the uploaded satellite image. The area primarily contains agricultural land, vegetation, and a small urban settlement. I've highlighted the detected regions below.",
  plan: ["spectral", "classification", "vlm"],
  tool_artifacts: {
    classification: {
      top_predictions: [
        { class_name: "Vegetation", confidence: 0.423 },
        { class_name: "Agricultural Land", confidence: 0.361 },
        { class_name: "Urban / Built-up", confidence: 0.124 },
        { class_name: "Water Bodies", confidence: 0.068 },
        { class_name: "Others", confidence: 0.024 }
      ]
    },
    spectral: {
      coverage_estimates: {
        dense_vegetation_percent: 42.3,
        water_body_percent: 6.8,
        builtup_percent: 12.4
      }
    }
  },
  history_length: 2
};
