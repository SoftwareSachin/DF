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

  const modelOptions = [
    { key: 'use_ensemble' as const, label: 'Ensemble Detection', description: 'Combined AI models for maximum accuracy' },
    { key: 'use_efficientnet' as const, label: 'EfficientNet-B0', description: 'Lightweight CNN with transfer learning' },
    { key: 'use_mobilenet' as const, label: 'MobileNet-V2', description: 'Mobile-optimized neural network' },
    { key: 'use_frequency' as const, label: 'Frequency Analysis', description: 'FFT-based digital artifact detection' },
    { key: 'use_face_analysis' as const, label: 'Face Analysis', description: 'Advanced facial landmark detection' },
  ];

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
      <div className="p-6 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Detection Settings</h3>
        <p className="text-sm text-gray-600 mt-1">Configure AI models and analysis parameters</p>
      </div>
      
      <div className="p-6 space-y-6">
        {/* AI Models Section */}
        <div>
          <h4 className="text-sm font-semibold text-gray-900 mb-4 flex items-center">
            <svg className="w-4 h-4 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            AI Models
          </h4>
          <div className="space-y-3">
            {modelOptions.map(({ key, label, description }) => (
              <div key={key} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="flex items-center h-5">
                  <input
                    type="checkbox"
                    checked={settings[key]}
                    onChange={(e) => updateSetting(key, e.target.checked)}
                    className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                  />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-gray-900">{label}</div>
                  <div className="text-xs text-gray-600">{description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Confidence Threshold */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <label className="text-sm font-semibold text-gray-900 flex items-center">
              <svg className="w-4 h-4 mr-2 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Confidence Threshold
            </label>
            <span className="text-sm font-medium text-blue-600 bg-blue-50 px-2 py-1 rounded">
              {(settings.confidence_threshold * 100).toFixed(0)}%
            </span>
          </div>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={settings.confidence_threshold}
            onChange={(e) => updateSetting('confidence_threshold', parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Conservative</span>
            <span>Balanced</span>
            <span>Aggressive</span>
          </div>
        </div>

        {/* Face Analysis Settings */}
        {settings.use_face_analysis && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-semibold text-gray-900 flex items-center">
                <svg className="w-4 h-4 mr-2 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Max Faces to Analyze
              </label>
              <span className="text-sm font-medium text-purple-600 bg-purple-50 px-2 py-1 rounded">
                {settings.max_faces}
              </span>
            </div>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={settings.max_faces}
              onChange={(e) => updateSetting('max_faces', parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
            />
          </div>
        )}

        {/* Video-specific settings */}
        {mediaType === 'video' && (
          <div className="space-y-4 pt-4 border-t border-gray-200">
            <h4 className="text-sm font-semibold text-gray-900 flex items-center">
              <svg className="w-4 h-4 mr-2 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              Video Analysis
            </h4>
            
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-gray-700">Frame Skip</label>
                <span className="text-sm font-medium text-red-600 bg-red-50 px-2 py-1 rounded">
                  Every {settings.frame_skip} frames
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="30"
                step="1"
                value={settings.frame_skip}
                onChange={(e) => updateSetting('frame_skip', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
            
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm font-medium text-gray-700">Max Frames</label>
                <span className="text-sm font-medium text-red-600 bg-red-50 px-2 py-1 rounded">
                  {settings.max_frames} frames
                </span>
              </div>
              <input
                type="range"
                min="10"
                max="300"
                step="10"
                value={settings.max_frames}
                onChange={(e) => updateSetting('max_frames', parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SettingsPanel;