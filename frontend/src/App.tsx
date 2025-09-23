import { useState } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import SettingsPanel from './components/SettingsPanel';
import ResultsDisplay from './components/ResultsDisplay';
import type { DetectionSettings, DetectionResponse, AnalysisProgress } from './types/api';
import { analyzeImage, analyzeVideo } from './services/api';
import './App.css';

const defaultSettings: DetectionSettings = {
  use_ensemble: true,
  use_efficientnet: true,
  use_mobilenet: true,
  use_frequency: true,
  use_face_analysis: true,
  confidence_threshold: 0.5,
  max_faces: 5,
  frame_skip: 5,
  max_frames: 50,
};

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mediaType, setMediaType] = useState<'image' | 'video'>('image');
  const [settings, setSettings] = useState<DetectionSettings>(defaultSettings);
  const [results, setResults] = useState<DetectionResponse | null>(null);
  const [progress, setProgress] = useState<AnalysisProgress>({
    uploading: false,
    analyzing: false,
    complete: false,
    error: null,
  });

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setResults(null);
    setProgress({ uploading: false, analyzing: false, complete: false, error: null });
    
    // Determine media type
    if (file.type.startsWith('video/')) {
      setMediaType('video');
    } else {
      setMediaType('image');
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setProgress({ uploading: true, analyzing: false, complete: false, error: null });

    try {
      setProgress({ uploading: false, analyzing: true, complete: false, error: null });
      
      let result: DetectionResponse;
      if (mediaType === 'video') {
        result = await analyzeVideo(selectedFile, settings);
      } else {
        result = await analyzeImage(selectedFile, settings);
      }

      setResults(result);
      setProgress({ uploading: false, analyzing: false, complete: true, error: null });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setProgress({ uploading: false, analyzing: false, complete: false, error: errorMessage });
    }
  };

  const getAcceptTypes = () => {
    return mediaType === 'video' 
      ? 'video/mp4,video/avi,video/mov,video/wmv'
      : 'image/jpeg,image/jpg,image/png,image/gif,image/bmp';
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Upload and Settings */}
          <div className="lg:col-span-1 space-y-6">
            {/* Media Type Toggle */}
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
                <button
                  onClick={() => setMediaType('image')}
                  className={`flex-1 py-2 px-4 text-sm font-medium rounded-md transition-colors ${
                    mediaType === 'image'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Image
                </button>
                <button
                  onClick={() => setMediaType('video')}
                  className={`flex-1 py-2 px-4 text-sm font-medium rounded-md transition-colors ${
                    mediaType === 'video'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  Video
                </button>
              </div>
            </div>

            {/* File Upload */}
            <FileUpload
              onFileSelect={handleFileSelect}
              isUploading={progress.uploading}
              accept={getAcceptTypes()}
            />

            {/* Selected File Info */}
            {selectedFile && (
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <h3 className="text-sm font-medium text-gray-900 mb-2">Selected File</h3>
                <div className="text-sm text-gray-600">
                  <p className="truncate">{selectedFile.name}</p>
                  <p>{(selectedFile.size / 1024 / 1024).toFixed(1)} MB</p>
                </div>
                <button
                  onClick={handleAnalyze}
                  disabled={progress.analyzing || progress.uploading}
                  className="w-full mt-4 bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {progress.analyzing ? 'Analyzing...' : 'Analyze'}
                </button>
              </div>
            )}

            {/* Settings Panel */}
            <SettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
              mediaType={mediaType}
            />
          </div>

          {/* Results */}
          <div className="lg:col-span-2">
            {progress.error && (
              <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex items-center">
                  <svg className="h-5 w-5 text-red-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-red-800 font-medium">Error: {progress.error}</span>
                </div>
              </div>
            )}
            
            <ResultsDisplay 
              results={results} 
              isAnalyzing={progress.analyzing || progress.uploading}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;