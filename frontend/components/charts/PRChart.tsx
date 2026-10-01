'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { cn } from '@/lib/utils';

interface PRChartProps {
  data: { recall: number; precision: number }[];
  auc: number;
  className?: string;
}

export function PRChart({ data, auc, className }: PRChartProps) {
  return (
    <div className={cn("w-full h-[300px]", className)}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 20, right: 20, left: 20, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
          <XAxis 
            dataKey="recall" 
            type="number" 
            domain={[0, 1]} 
            stroke="#94a3b8"
            label={{ value: 'Recall', position: 'bottom', fill: '#94a3b8' }}
          />
          <YAxis 
            type="number" 
            domain={[0, 1]} 
            stroke="#94a3b8"
            label={{ value: 'Precision', angle: -90, position: 'left', fill: '#94a3b8' }}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '0.5rem' }}
            formatter={(value: number) => value.toFixed(3)}
            labelFormatter={(label: number) => `Recall: ${label.toFixed(3)}`}
          />
          <Line 
            type="monotone" 
            dataKey="precision" 
            stroke="#60a5fa" 
            strokeWidth={3}
            dot={false}
            activeDot={{ r: 6, fill: '#60a5fa', stroke: '#020617' }}
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="text-center text-sm text-slate-300 mt-2">
        PR AUC (Average Precision): <span className="font-bold text-blue-400">{auc.toFixed(3)}</span>
      </div>
    </div>
  );
}
