import { useState, useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { Bot, User, X, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface ChatMessage {
  type: 'user' | 'ai';
  content: string;
  videoCommand?: {
    startTime: number;
    endTime?: number;
  };
}

interface ChatInterfaceProps {
  videoId: number;
  isOpen: boolean;
  onClose: () => void;
}

export default function ChatInterface({ videoId, isOpen, onClose }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      type: 'ai',
      content: "Hello! I've analyzed your video. Feel free to ask me anything about your performance, form, or specific reps. I can show you exactly what happened at any point in your workout!"
    }
  ]);
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const chatMutation = useMutation({
    mutationFn: async (message: string) => {
      const response = await apiRequest('POST', `/api/videos/${videoId}/chat`, { message });
      return response.json();
    },
    onSuccess: (data) => {
      setMessages(prev => [...prev, {
        type: 'ai',
        content: data.message,
        videoCommand: data.videoCommand
      }]);

      // Handle video command
      if (data.videoCommand) {
        const video = document.getElementById('analysisVideo') as HTMLVideoElement;
        if (video) {
          video.currentTime = data.videoCommand.startTime;
          video.play();
        }
      }
    }
  });

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    setMessages(prev => [...prev, { type: 'user', content: inputValue }]);
    chatMutation.mutate(inputValue);
    setInputValue("");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  const handleSuggestedQuestion = (question: string) => {
    setInputValue(question);
    handleSendMessage();
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className={`fixed right-0 top-0 h-full w-1/2 bg-white shadow-2xl transform transition-transform duration-300 z-50 ${
      isOpen ? 'translate-x-0' : 'translate-x-full'
    } ${isOpen ? 'block' : 'hidden'}`}>
      <div className="h-full flex flex-col">
        {/* Chat Header */}
        <div className="bg-primary p-6 text-white">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
                <Bot className="text-white h-5 w-5" />
              </div>
              <div>
                <h3 className="font-semibold">AI Fitness Coach</h3>
                <p className="text-sm text-blue-100">Ask me about your performance</p>
              </div>
            </div>
            <Button
              onClick={onClose}
              variant="ghost"
              size="sm"
              className="text-white hover:text-blue-200 hover:bg-white/20"
            >
              <X className="h-5 w-5" />
            </Button>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.map((message, index) => (
            <div key={index} className={`flex space-x-3 ${message.type === 'user' ? 'justify-end' : ''}`}>
              {message.type === 'ai' && (
                <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center">
                  <Bot className="text-white h-4 w-4" />
                </div>
              )}
              <div className={`flex-1 ${message.type === 'user' ? 'max-w-xs' : ''}`}>
                <div className={`rounded-xl p-4 ${
                  message.type === 'user' 
                    ? 'bg-primary text-white ml-auto' 
                    : 'bg-gray-100'
                }`}>
                  <p className={message.type === 'user' ? 'text-white' : 'text-gray-800'}>
                    {message.content}
                  </p>
                </div>
              </div>
              {message.type === 'user' && (
                <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                  <User className="text-gray-600 h-4 w-4" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="p-6 border-t border-gray-200">
          <div className="flex space-x-3">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about your performance..."
              className="flex-1"
            />
            <Button 
              onClick={handleSendMessage}
              disabled={chatMutation.isPending}
              className="bg-primary text-white hover:bg-blue-700"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex space-x-2 mt-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSuggestedQuestion("How was my form?")}
              className="bg-gray-100 text-gray-700 hover:bg-gray-200"
            >
              How was my form?
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSuggestedQuestion("Show me rep #3")}
              className="bg-gray-100 text-gray-700 hover:bg-gray-200"
            >
              Show me rep #3
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => handleSuggestedQuestion("What can I improve?")}
              className="bg-gray-100 text-gray-700 hover:bg-gray-200"
            >
              What can I improve?
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
