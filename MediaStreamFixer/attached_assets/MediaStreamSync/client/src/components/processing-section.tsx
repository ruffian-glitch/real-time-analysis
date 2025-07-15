import { useState, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Cog, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ProcessingSectionProps {
  selectedDrill: string;
  uploadedFile: File;
  onComplete: (videoId: number) => void;
}

export default function ProcessingSection({ selectedDrill, uploadedFile, onComplete }: ProcessingSectionProps) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("Initializing analysis...");
  const [useDemo, setUseDemo] = useState(false);

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiRequest('POST', '/api/videos/upload', formData);
      return response.json();
    },
    onSuccess: (data) => {
      setCurrentStep("Upload complete, starting AI analysis...");
      analyzeMutation.mutate(data.videoId);
    },
    onError: (error) => {
      console.error('Upload failed:', error);
      setCurrentStep("Upload failed. Please try again.");
    }
  });

  const analyzeMutation = useMutation({
    mutationFn: async (videoId: number) => {
      const response = await apiRequest('POST', `/api/videos/${videoId}/analyze`);
      return response.json();
    },
    onSuccess: (data) => {
      setCurrentStep("Analysis complete!");
      setTimeout(() => {
        onComplete(data.videoId);
      }, 1000);
    },
    onError: (error) => {
      console.error('Analysis failed:', error);
      setCurrentStep("Analysis failed. Please try uploading again.");
    }
  });

  const demoMutation = useMutation({
    mutationFn: async (drillType: string) => {
      const response = await apiRequest('POST', '/api/demo/analyze', {
        drillType
      });
      return response.json();
    },
    onSuccess: (data) => {
      setCurrentStep("Demo analysis complete!");
      setTimeout(() => {
        onComplete(data.videoId);
      }, 1000);
    },
    onError: (error) => {
      console.error('Demo failed:', error);
      setCurrentStep("Demo failed. Please try again.");
    }
  });

  useEffect(() => {
    const steps = [
      { progress: 20, text: 'Analyzing video orientation...' },
      { progress: 40, text: 'Detecting body landmarks...' },
      { progress: 60, text: 'Tracking movement patterns...' },
      { progress: 80, text: 'Calculating form metrics...' },
      { progress: 100, text: 'Generating insights...' }
    ];

    let stepIndex = 0;
    const interval = setInterval(() => {
      if (stepIndex < steps.length) {
        const step = steps[stepIndex];
        setProgress(step.progress);
        setCurrentStep(step.text);
        stepIndex++;
      } else {
        clearInterval(interval);
        // Start the actual upload and analysis
        const formData = new FormData();
        formData.append('video', uploadedFile);
        formData.append('drillType', selectedDrill);
        uploadMutation.mutate(formData);
      }
    }, 800);

    return () => clearInterval(interval);
  }, [selectedDrill, uploadedFile, uploadMutation]);

  return (
    <div className="mb-8">
      <div className="bg-white rounded-2xl shadow-lg p-8 text-center">
        <div className="animate-bounce mb-6">
          <Cog className="animate-spin text-6xl text-primary mx-auto h-24 w-24" />
        </div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Analyzing Your Performance</h3>
        <p className="text-lg text-gray-600 mb-6">Our AI is processing your video and analyzing your form...</p>
        <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
          <div 
            className="bg-primary h-3 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
        <p className="text-sm text-gray-500 mb-4">{currentStep}</p>
        
        {currentStep.includes("failed") && (
          <div className="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <p className="text-yellow-800 text-sm mb-4">
              Having trouble with your video? Try our demo mode to see how the AI analysis works!
            </p>
            <Button 
              onClick={() => {
                setCurrentStep("Starting AI demo analysis...");
                demoMutation.mutate(selectedDrill);
              }}
              disabled={demoMutation.isPending}
              className="flex items-center gap-2"
            >
              <Zap className="h-4 w-4" />
              Try AI Demo Analysis
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
