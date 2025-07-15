import { useQuery } from "@tanstack/react-query";
import { Eye } from "lucide-react";
import PerformanceChart from "./performance-chart";

interface ResultsSectionProps {
  videoId: number;
  selectedDrill: string;
  chatOpen: boolean;
}

export default function ResultsSection({ videoId, selectedDrill, chatOpen }: ResultsSectionProps) {
  const { data: analysisData, isLoading } = useQuery({
    queryKey: ['/api/videos', videoId, 'analysis'],
    enabled: !!videoId
  });

  if (isLoading) {
    return <div className="text-center py-8">Loading analysis...</div>;
  }

  if (!analysisData) {
    return <div className="text-center py-8">No analysis data available</div>;
  }

  const analysis = analysisData.analysisData;

  return (
    <div>
      <div className={`grid grid-cols-1 ${chatOpen ? 'lg:grid-cols-1' : 'lg:grid-cols-2'} gap-8 transition-all duration-300`}>
        {/* Video Player Section */}
        <div className={`${chatOpen ? 'lg:col-span-1' : 'lg:col-span-1'} transition-all duration-300`}>
          <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-xl font-bold text-gray-900 mb-2">Your Performance Video</h3>
              <p className="text-gray-600">{selectedDrill.charAt(0).toUpperCase() + selectedDrill.slice(1)} Analysis</p>
            </div>
            <div className="p-6">
              <div className="relative bg-gray-900 rounded-xl overflow-hidden">
                <video id="analysisVideo" className="w-full h-64 object-cover" controls>
                  <source src="https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4" type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
                <div className="absolute bottom-4 left-4 bg-black/70 text-white px-3 py-1 rounded-full text-sm flex items-center">
                  <Eye className="h-4 w-4 mr-1" />
                  AI Analysis Active
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Metrics Section */}
        {!chatOpen && (
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Performance Metrics</h3>
              
              {/* Key Stats */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="bg-gradient-to-r from-primary/10 to-blue-100 p-4 rounded-xl">
                  <div className="text-2xl font-bold text-primary">{analysis.totalReps}</div>
                  <div className="text-sm text-gray-600">Total Reps</div>
                </div>
                <div className="bg-gradient-to-r from-secondary/10 to-green-100 p-4 rounded-xl">
                  <div className="text-2xl font-bold text-secondary">{analysis.formScore}/10</div>
                  <div className="text-sm text-gray-600">Form Score</div>
                </div>
                <div className="bg-gradient-to-r from-accent/10 to-yellow-100 p-4 rounded-xl">
                  <div className="text-2xl font-bold text-accent">{Math.floor(analysis.totalTime / 60)}:{(analysis.totalTime % 60).toString().padStart(2, '0')}</div>
                  <div className="text-sm text-gray-600">Time Under Tension</div>
                </div>
                <div className="bg-gradient-to-r from-purple-100 to-pink-100 p-4 rounded-xl">
                  <div className="text-2xl font-bold text-purple-600">{analysis.consistency}%</div>
                  <div className="text-sm text-gray-600">Consistency</div>
                </div>
              </div>

              {/* Performance Chart */}
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Rep Quality Over Time</h4>
                <PerformanceChart data={analysis.reps} />
              </div>

              {/* Detailed Analysis */}
              <div>
                <h4 className="text-lg font-semibold text-gray-900 mb-4">Detailed Analysis</h4>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-secondary rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-white text-xs">✓</span>
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Excellent form control</div>
                      <div className="text-sm text-gray-600">Maintained consistent angles throughout</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-accent rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-white text-xs">!</span>
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Minor variations in later reps</div>
                      <div className="text-sm text-gray-600">Consider pacing yourself for consistency</div>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <div className="w-6 h-6 bg-primary rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-white text-xs">i</span>
                    </div>
                    <div>
                      <div className="font-medium text-gray-900">Strong finish</div>
                      <div className="text-sm text-gray-600">Final reps showed good recovery</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Rep-by-Rep Breakdown */}
      <div className="mt-8 bg-white rounded-2xl shadow-lg p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-6">Rep-by-Rep Breakdown</h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Rep</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Time</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Form Score</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Angle</th>
                <th className="text-left py-3 px-4 font-semibold text-gray-900">Notes</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {analysis.reps.map((rep: any, index: number) => (
                <tr 
                  key={index} 
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    const video = document.getElementById('analysisVideo') as HTMLVideoElement;
                    if (video) {
                      video.currentTime = rep.start_frame / 30; // Assuming 30fps
                      video.play();
                    }
                  }}
                >
                  <td className="py-3 px-4 font-medium">{rep.rep_number}</td>
                  <td className="py-3 px-4 text-gray-600">{Math.floor(rep.start_frame / 30)}s</td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-1 rounded-full text-sm ${
                      rep.form_score >= 8.5 ? 'bg-green-100 text-green-800' : 
                      rep.form_score >= 7.5 ? 'bg-yellow-100 text-yellow-800' : 
                      'bg-red-100 text-red-800'
                    }`}>
                      {rep.form_score}/10
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-600">
                    {rep.elbow_angle ? `${rep.elbow_angle}°` : 
                     rep.knee_angle ? `${rep.knee_angle}°` : 
                     'N/A'}
                  </td>
                  <td className="py-3 px-4 text-gray-600">{rep.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
