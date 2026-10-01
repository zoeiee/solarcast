'use client';

import * as React from 'react';
import { UploadCloud, File, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from './Button';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  className?: string;
}

export function UploadZone({ onFileSelect, accept = ".png,.jpg,.jpeg,.fits,.npy", className }: UploadZoneProps) {
  const [isDragging, setIsDragging] = React.useState(false);
  const [file, setFile] = React.useState<File | null>(null);
  const fileInputRef = React.useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const selectedFile = e.dataTransfer.files[0];
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  };

  const clearFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  return (
    <div
      className={cn(
        "relative flex flex-col items-center justify-center w-full min-h-[200px] border-2 border-dashed rounded-xl transition-colors cursor-pointer",
        isDragging ? "border-amber-500 bg-amber-500/10" : "border-slate-700 bg-slate-800/30 hover:bg-slate-800/50",
        className
      )}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleChange}
        accept={accept}
        className="hidden"
      />
      
      {file ? (
        <div className="flex flex-col items-center p-4">
          <div className="p-3 bg-slate-800 rounded-full mb-3 relative group">
            <File className="w-8 h-8 text-amber-500" />
            <button 
              onClick={clearFile}
              className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full text-white opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
          <p className="text-sm font-medium text-slate-200">{file.name}</p>
          <p className="text-xs text-slate-400 mt-1">
            {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center p-6 text-center">
          <UploadCloud className="w-10 h-10 text-slate-400 mb-4" />
          <p className="text-sm text-slate-200 font-medium mb-1">
            Click to upload or drag and drop
          </p>
          <p className="text-xs text-slate-400">
            Supports PNG, JPG, FITS, NPY
          </p>
        </div>
      )}
    </div>
  );
}
