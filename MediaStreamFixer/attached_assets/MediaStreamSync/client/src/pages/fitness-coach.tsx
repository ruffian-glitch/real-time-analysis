import { useState, useEffect } from "react";
import { Dumbbell, Bell, User } from "lucide-react";
import DrillSelection from "@/components/drill-selection";
import VideoUpload from "@/components/video-upload";
import ProcessingSection from "@/components/processing-section";
import ResultsSection from "@/components/results-section";
import ChatInterface from "@/components/chat-interface";
import { Button } from "@/components/ui/button";
import { MessageCircle } from "lucide-react";

type Section = "upload" | "processing" | "results";

export default function FitnessCoach() {
  const [currentSection, setCurrentSection] = useState<Section>("upload");
  const [selectedDrill, setSelectedDrill] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [videoId, setVideoId] = useState<number | null>(null);
  const [chatOpen, setChatOpen] = useState(false);

  const handleAnalyze = () => {
    setCurrentSection("processing");
  };

  const handleProcessingComplete = (id: number) => {
    setVideoId(id);
    setCurrentSection("results");
  };

  const openChat = () => {
    setChatOpen(true);
  };

  const closeChat = () => {
    setChatOpen(false);
  };

  return (
    <div className="bg-gray-50 min-h-screen">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary rounded-xl flex items-center justify-center">
                <Dumbbell className="text-white text-lg" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">AI Fitness Coach</h1>
                <p className="text-sm text-gray-500">Analyze Your Performance</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="ghost" size="sm" className="text-gray-500 hover:text-gray-700">
                <Bell className="h-5 w-5" />
              </Button>
              <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                <User className="text-gray-600 h-4 w-4" />
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentSection === "upload" && (
          <div className="mb-8">
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">Upload Your Fitness Video</h2>
                <p className="text-lg text-gray-600">Select your drill type and upload a video to get instant AI-powered analysis</p>
              </div>

              <DrillSelection 
                selectedDrill={selectedDrill} 
                onDrillSelect={setSelectedDrill} 
              />

              <VideoUpload 
                uploadedFile={uploadedFile} 
                onFileUpload={setUploadedFile} 
              />

              <div className="text-center">
                <Button 
                  onClick={handleAnalyze}
                  disabled={!selectedDrill || !uploadedFile}
                  className="bg-primary text-white px-8 py-4 rounded-xl font-semibold text-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
                >
                  <span className="mr-2">🧠</span>
                  Analyze Video
                </Button>
              </div>
            </div>
          </div>
        )}

        {currentSection === "processing" && (
          <ProcessingSection 
            selectedDrill={selectedDrill!}
            uploadedFile={uploadedFile!}
            onComplete={handleProcessingComplete}
          />
        )}

        {currentSection === "results" && videoId && (
          <ResultsSection 
            videoId={videoId}
            selectedDrill={selectedDrill!}
            chatOpen={chatOpen}
          />
        )}
      </main>

      {/* Chat Interface */}
      {currentSection === "results" && videoId && (
        <ChatInterface 
          videoId={videoId}
          isOpen={chatOpen}
          onClose={closeChat}
        />
      )}

      {/* Chat Button */}
      {currentSection === "results" && !chatOpen && (
        <Button
          onClick={openChat}
          className="fixed bottom-6 right-6 w-16 h-16 bg-primary text-white rounded-full shadow-lg hover:bg-blue-700 hover:scale-110 transition-all duration-200 z-40"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      )}
    </div>
  );
}
