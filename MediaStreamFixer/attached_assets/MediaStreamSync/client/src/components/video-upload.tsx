import { CloudUpload } from "lucide-react";
import { Button } from "@/components/ui/button";

interface VideoUploadProps {
  uploadedFile: File | null;
  onFileUpload: (file: File) => void;
}

export default function VideoUpload({ uploadedFile, onFileUpload }: VideoUploadProps) {
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  return (
    <div className="mb-8">
      <label className="block text-lg font-semibold text-gray-900 mb-4">Upload Video</label>
      <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-primary transition-colors">
        {uploadedFile ? (
          <div className="animate-fade-in">
            <div className="text-4xl text-green-500 mb-4">📹</div>
            <p className="text-lg text-gray-800 mb-2">{uploadedFile.name}</p>
            <p className="text-sm text-gray-500">{(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
          </div>
        ) : (
          <>
            <CloudUpload className="text-4xl text-gray-400 mb-4 mx-auto h-16 w-16" />
            <p className="text-lg text-gray-600 mb-2">Drag and drop your video file here</p>
            <p className="text-sm text-gray-500 mb-4">or click to browse files</p>
            <input 
              type="file" 
              accept="video/*" 
              onChange={handleFileSelect}
              className="hidden"
              id="video-upload"
            />
            <Button 
              onClick={() => document.getElementById('video-upload')?.click()}
              className="bg-primary text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 transition-colors"
            >
              Choose File
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
