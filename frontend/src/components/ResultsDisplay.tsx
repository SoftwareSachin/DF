import React from 'react';
import type { DetectionResponse } from '../types/api';

interface ResultsDisplayProps {
  results: DetectionResponse | null;
  isAnalyzing: boolean;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, isAnalyzing }) => {
  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <div className="flex items-center justify-center space-x-2">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="text-gray-600">Analyzing...</span>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="bg-gray-50 rounded-lg border-2 border-dashed border-gray-300 p-8">
        <div className="text-center text-gray-500">
          <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p>Upload and analyze a file to see results</p>
        </div>
      </div>
    );
  }

  if (!results.success) {
    return (
      <div className="bg-white rounded-lg border border-red-200 p-6">
        <div className="flex items-center space-x-2 text-red-600 mb-4">
          <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="font-medium">Analysis Failed</span>
        </div>
        <p className="text-red-600">{results.error}</p>
      </div>
    );
  }

  const { summary, predictions, meta } = results;

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className={`bg-white rounded-lg border-2 p-6 ${
        summary.is_fake ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'
      }`}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Detection Summary</h3>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            summary.is_fake 
              ? 'bg-red-100 text-red-800' 
              : 'bg-green-100 text-green-800'
          }`}>
            {summary.is_fake ? 'Likely Fake' : 'Likely Real'}
          </div>
        </div>
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Confidence Score:</span>
            <span className="ml-2 font-medium">{(summary.confidence_score * 100).toFixed(1)}%</span>
          </div>
          <div>
            <span className="text-gray-600">Processing Time:</span>
            <span className="ml-2 font-medium">{summary.total_processing_time.toFixed(2)}s</span>
          </div>
        </div>

        {/* Confidence Bar */}
        <div className="mt-4">
          <div className="flex justify-between text-xs text-gray-600 mb-1">
            <span>Real</span>
            <span>Fake</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full ${summary.is_fake ? 'bg-red-500' : 'bg-green-500'}`}
              style={{ width: `${summary.confidence_score * 100}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Media Info */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Media Information</h3>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Format:</span>
            <span className="ml-2 font-medium">{meta.format}</span>
          </div>
          <div>
            <span className="text-gray-600">Size:</span>
            <span className="ml-2 font-medium">{(meta.size_bytes / 1024 / 1024).toFixed(1)} MB</span>
          </div>
          <div>
            <span className="text-gray-600">Dimensions:</span>
            <span className="ml-2 font-medium">{meta.width} × {meta.height}</span>
          </div>
        </div>
      </div>

      {/* Model Results */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">Model Results</h3>
        <div className="space-y-4">
          {Object.entries(predictions).map(([modelName, prediction]) => {
            if (!prediction) return null;
            
            const displayName = modelName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            
            return (
              <div key={modelName} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <span className="font-medium text-gray-900">{displayName}</span>
                  <span className="ml-2 text-sm text-gray-600">
                    ({prediction.processing_time.toFixed(2)}s)
                  </span>
                </div>
                <div className="flex items-center space-x-3">
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    prediction.is_fake
                      ? 'bg-red-100 text-red-800'
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {prediction.is_fake ? 'Fake' : 'Real'}
                  </div>
                  <span className="text-sm font-medium text-gray-900">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;