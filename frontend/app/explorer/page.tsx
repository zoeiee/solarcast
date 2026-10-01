'use client';

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { MagnetogramViewer } from '@/components/magnetogram/MagnetogramViewer';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Search } from 'lucide-react';

export default function ExplorerPage() {
  // Mock histogram data for the magnetogram pixel intensities
  const histogramData = Array.from({ length: 40 }, (_, i) => ({
    bin: (i - 20) * 100, // -2000 to +2000 Gauss approx
    count: Math.pow(2.718, -Math.pow((i - 20) / 5, 2)) * 10000 + Math.random() * 500
  }));

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
          <Search className="w-8 h-8 text-amber-500" /> Interactive Explorer
        </h1>
        <p className="text-slate-400">Inspect magnetogram details and magnetic field distributions.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Magnetogram View</CardTitle>
          </CardHeader>
          <CardContent>
            <MagnetogramViewer />
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Field Intensity Distribution</CardTitle>
              <CardDescription>Pixel count vs Field Strength (Gauss)</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={histogramData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} vertical={false} />
                    <XAxis dataKey="bin" stroke="#94a3b8" fontSize={12} tickFormatter={(val) => val === 0 ? '0' : val} />
                    <YAxis stroke="#94a3b8" fontSize={12} width={40} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }}
                      labelFormatter={(label) => `Gauss: ~${label}`}
                    />
                    <Bar dataKey="count" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Region Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-slate-400">Max Positive Field:</span>
                  <span className="font-mono text-slate-200">+2,450 G</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Min Negative Field:</span>
                  <span className="font-mono text-slate-200">-2,810 G</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Net Flux:</span>
                  <span className="font-mono text-slate-200">1.2e22 Mx</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Active Area:</span>
                  <span className="font-mono text-slate-200">1,450 MSH</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
