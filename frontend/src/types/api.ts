export interface DetectionSettings {
  use_ensemble: boolean;
  use_efficientnet: boolean;
  use_mobilenet: boolean;
  use_frequency: boolean;
  use_face_analysis: boolean;
  confidence_threshold: number;
  max_faces: number;
  frame_skip: number;
  max_frames: number;
}

export interface MediaMetadata {
  width: number;
  height: number;
  format: string;
  size_bytes: number;
}

export interface ModelPrediction {
  confidence: number;
  is_fake: boolean;
  processing_time: number;
}

export interface DetectionPredictions {
  real_ai_openai?: ModelPrediction;
  ensemble?: ModelPrediction;
  efficientnet?: ModelPrediction;
  mobilenet?: ModelPrediction;
  frequency?: ModelPrediction;
  face?: ModelPrediction;
}

export interface DetectionSummary {
  is_fake: boolean;
  confidence_score: number;
  total_processing_time: number;
}

export interface DetectionResponse {
  success: boolean;
  meta: MediaMetadata;
  predictions: DetectionPredictions;
  summary: DetectionSummary;
  error?: string;
}

export interface AnalysisProgress {
  uploading: boolean;
  analyzing: boolean;
  complete: boolean;
  error: string | null;
}