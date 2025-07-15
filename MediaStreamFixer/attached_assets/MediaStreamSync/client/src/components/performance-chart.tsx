import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

interface PerformanceChartProps {
  data: Array<{
    rep_number: number;
    form_score: number;
  }>;
}

export default function PerformanceChart({ data }: PerformanceChartProps) {
  const chartData = data.map(rep => ({
    rep: `Rep ${rep.rep_number}`,
    score: rep.form_score
  }));

  return (
    <div className="w-full h-48">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <XAxis 
            dataKey="rep" 
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#6B7280' }}
          />
          <YAxis 
            domain={[0, 10]}
            axisLine={false}
            tickLine={false}
            tick={{ fontSize: 12, fill: '#6B7280' }}
          />
          <Tooltip 
            contentStyle={{
              backgroundColor: '#1F2937',
              border: 'none',
              borderRadius: '8px',
              color: 'white'
            }}
          />
          <Line 
            type="monotone" 
            dataKey="score" 
            stroke="hsl(207, 90%, 54%)" 
            strokeWidth={3}
            dot={{ fill: 'hsl(207, 90%, 54%)', strokeWidth: 2, r: 4 }}
            activeDot={{ r: 6, fill: 'hsl(207, 90%, 54%)' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
