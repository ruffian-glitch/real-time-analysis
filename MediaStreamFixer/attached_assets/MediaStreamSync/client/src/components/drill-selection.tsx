import { DrillTypes } from "@shared/schema";
import { ArrowDown, ArrowUp, Clock, Minus, Scale } from "lucide-react";

interface DrillSelectionProps {
  selectedDrill: string | null;
  onDrillSelect: (drill: string) => void;
}

const drillOptions = [
  { id: DrillTypes.PUSH_UPS, label: "Push-ups", icon: ArrowDown },
  { id: DrillTypes.SQUATS, label: "Squats", icon: ArrowUp },
  { id: DrillTypes.SIT_UPS, label: "Sit-ups", icon: ArrowUp },
  { id: DrillTypes.WALL_SIT, label: "Wall Sit", icon: Clock },
  { id: DrillTypes.ELBOW_PLANK, label: "Elbow Plank", icon: Minus },
  { id: DrillTypes.SINGLE_LEG_BALANCE, label: "Single Leg Balance", icon: Scale },
];

export default function DrillSelection({ selectedDrill, onDrillSelect }: DrillSelectionProps) {
  return (
    <div className="mb-8">
      <label className="block text-lg font-semibold text-gray-900 mb-4">Select Drill Type</label>
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {drillOptions.map((drill) => {
          const Icon = drill.icon;
          const isSelected = selectedDrill === drill.id;
          
          return (
            <button
              key={drill.id}
              onClick={() => onDrillSelect(drill.id)}
              className={`p-4 border-2 rounded-xl hover:border-primary hover:bg-primary/5 transition-all duration-200 text-center ${
                isSelected 
                  ? 'border-primary bg-primary/5' 
                  : 'border-gray-200'
              }`}
            >
              <Icon className="text-2xl text-gray-600 mb-2 mx-auto" />
              <div className="font-medium text-gray-900">{drill.label}</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
