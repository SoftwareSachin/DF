import React, { useCallback } from 'react';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  isUploading: boolean;
  accept: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileSelect, isUploading, accept }) => {
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
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
    <div className="bg-white rounded-lg border-2 border-dashed border-gray-300 p-8">
      <div
        className="text-center"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
          <svg fill="none" stroke="currentColor" viewBox="0 0 48 48">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
            />
          </svg>
        </div>
        <div className="text-sm text-gray-600">
          <label className="cursor-pointer">
            <span className="font-medium text-blue-600 hover:text-blue-500">
              Upload a file
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
        <p className="text-xs text-gray-500 mt-1">
          {accept.includes('image') ? 'PNG, JPG, GIF up to 100MB' : 'MP4, AVI, MOV up to 100MB'}
        </p>
      </div>
    </div>
  );
};

export default FileUpload;