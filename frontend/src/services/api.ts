import axios from 'axios';
import type { DetectionSettings, DetectionResponse } from '../types/api';

const API_BASE = '';

const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000, // 2 minutes timeout for analysis
});

export const analyzeImage = async (
  file: File,
  settings: DetectionSettings
): Promise<DetectionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('settings', JSON.stringify(settings));

  const response = await apiClient.post<DetectionResponse>(
    '/api/analyze/image',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

export const analyzeVideo = async (
  file: File,
  settings: DetectionSettings
): Promise<DetectionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('settings', JSON.stringify(settings));

  const response = await apiClient.post<DetectionResponse>(
    '/api/analyze/video',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

export const getHealthStatus = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

export const getModelsInfo = async () => {
  const response = await apiClient.get('/api/models');
  return response.data;
};