import React from 'react';
import type { DetectionSettings } from '../types/api';

interface SettingsPanelProps {
  settings: DetectionSettings;
  onSettingsChange: (settings: DetectionSettings) => void;
  mediaType: 'image' | 'video';
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ 
  settings, 
  onSettingsChange, 
  mediaType 
}) => {
  const updateSetting = (key: keyof DetectionSettings, value: any) => {
    onSettingsChange({ ...settings, [key]: value });
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Detection Settings</h3>
      
      <div className="space-y-4">
        {/* Model Selection */}
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-2">AI Models</h4>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.use_ensemble}
                onChange={(e) => updateSetting('use_ensemble', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-600">Ensemble Detection</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.use_efficientnet}
                onChange={(e) => updateSetting('use_efficientnet', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-600">EfficientNet-B0</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.use_mobilenet}
                onChange={(e) => updateSetting('use_mobilenet', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-600">MobileNet-V2</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.use_frequency}
                onChange={(e) => updateSetting('use_frequency', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-600">Frequency Analysis</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={settings.use_face_analysis}
                onChange={(e) => updateSetting('use_face_analysis', e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-600">Face Analysis</span>
            </label>
          </div>
        </div>

        {/* Confidence Threshold */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Confidence Threshold: {settings.confidence_threshold.toFixed(1)}
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.1"
            value={settings.confidence_threshold}
            onChange={(e) => updateSetting('confidence_threshold', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Face Analysis Settings */}
        {settings.use_face_analysis && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Faces: {settings.max_faces}
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={settings.max_faces}
              onChange={(e) => updateSetting('max_faces', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        )}

        {/* Video-specific settings */}
        {mediaType === 'video' && (
          <>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Frame Skip: {settings.frame_skip}
              </label>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={settings.frame_skip}
                onChange={(e) => updateSetting('frame_skip', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Max Frames: {settings.max_frames}
              </label>
              <input
                type="range"
                min="10"
                max="300"
                step="10"
                value={settings.max_frames}
                onChange={(e) => updateSetting('max_frames', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default SettingsPanel;