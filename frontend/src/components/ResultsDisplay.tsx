import React from 'react';
import type { DetectionResponse } from '../types/api';

interface ResultsDisplayProps {
  results: DetectionResponse | null;
  isAnalyzing: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isAnalyzing }) => {
  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-8">
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="relative">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-blue-200 border-t-blue-600"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-4 h-4 bg-blue-600 rounded-full animate-pulse"></div>
            </div>
          </div>
          <div className="text-center">
            <h3 className="text-lg font-semibold text-gray-900">Analyzing Media</h3>
            <p className="text-sm text-gray-600 mt-1">Processing with AI detection models...</p>
          </div>
          <div className="flex space-x-2">
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-gray-50 rounded-xl border-2 border-dashed border-gray-300 p-12">
        <div className="text-center">
          <div className="mx-auto w-16 h-16 text-gray-400 mb-6">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Ready for Analysis</h3>
          <p className="text-gray-600">Upload an image or video to begin AI-powered deepfake detection</p>
        </div>
      </div>
    );
  }

  if (!results.success) {
    return (
      <div className="bg-white rounded-xl border border-red-200 shadow-sm overflow-hidden">
        <div className="bg-red-50 px-6 py-4 border-b border-red-200">
          <div className="flex items-center space-x-3">
            <div className="bg-red-100 p-2 rounded-full">
              <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-red-900">Analysis Failed</h3>
              <p className="text-sm text-red-700">An error occurred during processing</p>
            </div>
          </div>
        </div>
        <div className="p-6">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800 font-mono text-sm">{results.error}</p>
          </div>
        </div>
      </div>
    );
  }

  const { summary, predictions, meta } = results;
  const confidencePercentage = summary.confidence_score * 100;

  return (
    <div className="space-y-6">
      {/* Main Detection Result */}
      <div className={`bg-white rounded-xl border-2 shadow-lg overflow-hidden ${
        summary.is_fake ? 'border-red-200' : 'border-green-200'
      }`}>
        <div className={`px-6 py-4 ${
          summary.is_fake ? 'bg-red-50 border-b border-red-200' : 'bg-green-50 border-b border-green-200'
        }`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className={`p-3 rounded-full ${
                summary.is_fake ? 'bg-red-100' : 'bg-green-100'
              }`}>
                {summary.is_fake ? (
                  <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.872-.833-2.464 0L3.35 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                ) : (
                  <svg className="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
              </div>
              <div>
                <h3 className="text-2xl font-bold text-gray-900">
                  {summary.is_fake ? 'Potentially Fake' : 'Likely Authentic'}
                </h3>
                <p className={`text-sm font-medium ${
                  summary.is_fake ? 'text-red-700' : 'text-green-700'
                }`}>
                  {confidencePercentage.toFixed(1)}% confidence • {summary.total_processing_time.toFixed(2)}s processing
                </p>
              </div>
            </div>
            <div className={`px-4 py-2 rounded-full text-sm font-bold ${
              summary.is_fake 
                ? 'bg-red-100 text-red-800' 
                : 'bg-green-100 text-green-800'
            }`}>
              {summary.is_fake ? 'FAKE DETECTED' : 'AUTHENTIC'}
            </div>
          </div>
        </div>

        <div className="p-6">
          {/* Confidence Visualization */}
          <div className="mb-6">
            <div className="flex justify-between text-sm font-medium text-gray-700 mb-2">
              <span>Authenticity Score</span>
              <span>{confidencePercentage.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <div 
                className={`h-3 rounded-full transition-all duration-1000 ease-out ${
                  summary.is_fake ? 'bg-red-500' : 'bg-green-500'
                }`}
                style={{ width: `${confidencePercentage}%` }}
              ></div>
            </div>
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Authentic</span>
              <span>Suspicious</span>
              <span>Fake</span>
            </div>
          </div>

          {/* Media Information */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{meta.format}</div>
              <div className="text-sm text-gray-600">Format</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{(meta.size_bytes / 1024 / 1024).toFixed(1)}MB</div>
              <div className="text-sm text-gray-600">File Size</div>
            </div>
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{meta.width}×{meta.height}</div>
              <div className="text-sm text-gray-600">Resolution</div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Results */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 flex items-center">
            <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
            Individual Model Results
          </h3>
          <p className="text-sm text-gray-600 mt-1">Detailed analysis from each AI detection model</p>
        </div>
        <div className="p-6">
          <div className="grid gap-4">
            {Object.entries(predictions).map(([modelName, prediction]) => {
              if (!prediction) return null;
              
              const displayName = modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
              const modelConfidence = prediction.confidence * 100;
              
              return (
                <div key={modelName} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold text-gray-900">{displayName}</h4>
                      <p className="text-sm text-gray-600">Processing time: {prediction.processing_time.toFixed(3)}s</p>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                        prediction.is_fake
                          ? 'bg-red-100 text-red-800'
                          : 'bg-green-100 text-green-800'
                      }`}>
                        {prediction.is_fake ? 'Fake' : 'Real'}
                      </div>
                      <span className="text-lg font-bold text-gray-900">
                        {modelConfidence.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        prediction.is_fake ? 'bg-red-400' : 'bg-green-400'
                      }`}
                      style={{ width: `${modelConfidence}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;