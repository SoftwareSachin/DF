import React, { useCallback, useState } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isUploading: boolean;
  accept: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isUploading, accept }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      onFileSelect(files[0]);
    }
  }, [onFileSelect]);

  return (
    <div className={`relative bg-white rounded-xl border-2 border-dashed transition-all duration-200 ${
      isDragOver 
        ? 'border-blue-500 bg-blue-50' 
        : 'border-gray-300 hover:border-gray-400'
    } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}>
      <div
        className="p-12 text-center"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="mx-auto w-16 h-16 text-gray-400 mb-6">
          <svg fill="none" stroke="currentColor" viewBox="0 0 48 48">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            />
          </svg>
        </div>
        
        <div className="space-y-2">
          <div className="text-xl font-semibold text-gray-900">
            {isDragOver ? 'Drop your file here' : 'Upload media for analysis'}
          </div>
          <div className="text-gray-600">
            <label className="cursor-pointer">
              <span className="font-medium text-blue-600 hover:text-blue-700 transition-colors">
                Browse files
              </span>
              <input
                type="file"
                className="sr-only"
                accept={accept}
                onChange={handleFileInput}
                disabled={isUploading}
              />
            </label>
            <span> or drag and drop</span>
          </div>
          <p className="text-sm text-gray-500">
            {accept.includes('image') 
              ? 'Supports: PNG, JPG, GIF, BMP (up to 100MB)' 
              : 'Supports: MP4, AVI, MOV, WMV (up to 100MB)'
            }
          </p>
        </div>

        {isUploading && (
          <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-90 rounded-xl">
            <div className="flex items-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              <span className="text-gray-600 font-medium">Uploading...</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;