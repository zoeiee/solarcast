'use client';

import React, { useState } from 'react';
import { UploadZone } from '@/components/ui/UploadZone';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { GradCAMOverlay } from '@/components/magnetogram/GradCAMOverlay';
import { predictFlare, explainPrediction } from '@/lib/api';
import { ExplanationResult } from '@/lib/types';
import { formatProbability, getRiskLevel, formatMetricValue } from '@/lib/utils';
import { AlertCircle, Zap, Activity } from 'lucide-react';

export default function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ExplanationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    try {
      // Simulate real process: user uploads image -> predict -> explain
      // For a demo/mock UI with real backend, we'll just hit the explain endpoint 
      // which should return everything (prediction + heatmap).
      const expRes = await explainPrediction(file);
      setResult(expRes);
    } catch (err) {
      console.error(err);
      setError('Failed to analyze the magnetogram. Please try again or ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoClick = async (demoName: string) => {
    // In a real app, fetch demo file. Here we create a mock file to pass the check.
    const mockFile = new File(["dummy content"], `${demoName}.png`, { type: "image/png" });
    setFile(mockFile);
  };

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
          <Zap className="w-8 h-8 text-amber-500" /> Analyze Magnetogram
        </h1>
        <p className="text-slate-400">Upload a solar magnetogram to estimate the probability of an M or X-class flare within 24 hours.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-1 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Input Data</CardTitle>
              <CardDescription>Upload FITS, NPY, PNG, or JPG</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <UploadZone onFileSelect={setFile} />
              
              <Button 
                className="w-full" 
                size="lg" 
                disabled={!file || loading}
                onClick={handleAnalyze}
                isLoading={loading}
              >
                {loading ? 'Analyzing...' : 'Run Analysis'}
              </Button>

              {error && (
                <div className="p-3 bg-red-900/30 border border-red-900 text-red-300 rounded-md text-sm flex gap-2 items-start">
                  <AlertCircle className="w-5 h-5 shrink-0" />
                  <p>{error}</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Sample Regions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-2">
                {['AR12673', 'AR12192', 'QuietSun', 'AR12992'].map((demo) => (
                  <button
                    key={demo}
                    onClick={() => handleDemoClick(demo)}
                    className="aspect-square bg-slate-900 border border-slate-700 rounded-md hover:border-amber-500 transition-colors flex items-center justify-center relative overflow-hidden group"
                  >
                    <div className="absolute inset-0 bg-gradient-to-tr from-slate-950 via-slate-800 to-slate-900 opacity-80 group-hover:opacity-100 transition-opacity" />
                    <span className="relative text-xs font-medium text-slate-300 group-hover:text-amber-400">{demo}</span>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          {!result && !loading && (
            <div className="h-full min-h-[400px] border border-dashed border-slate-700 rounded-xl flex flex-col items-center justify-center text-slate-500 space-y-4">
              <Activity className="w-12 h-12 opacity-50" />
              <p>Upload a magnetogram and run analysis to see results</p>
            </div>
          )}

          {loading && (
            <div className="h-full min-h-[400px] border border-slate-800 bg-slate-900/50 rounded-xl flex flex-col items-center justify-center text-slate-400 space-y-6">
              <div className="w-16 h-16 relative flex items-center justify-center">
                <div className="absolute inset-0 border-4 border-slate-800 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-amber-500 rounded-full border-t-transparent animate-spin"></div>
              </div>
              <div className="text-center space-y-1">
                <p className="font-medium text-slate-200">Processing Magnetogram</p>
                <p className="text-sm">Extracting features and generating Grad-CAM maps...</p>
              </div>
            </div>
          )}

          {result && !loading && (
            <div className="space-y-6 animate-in fade-in duration-500">
              <Card className="border-amber-500/20 bg-amber-500/5">
                <CardContent className="p-6 md:p-8 flex flex-col md:flex-row items-center justify-between gap-6">
                  <div className="space-y-2 text-center md:text-left">
                    <h2 className="text-lg font-medium text-slate-300 uppercase tracking-wider">24h Flare Probability</h2>
                    <div className="text-5xl font-bold text-slate-100">
                      {formatProbability(result.probability)}
                    </div>
                    <div className="flex items-center gap-2 justify-center md:justify-start pt-2">
                      <Badge variant={getRiskLevel(result.probability, result.threshold).color as any}>
                        {getRiskLevel(result.probability, result.threshold).level}
                      </Badge>
                      <span className="text-sm text-slate-400">
                        Threshold: {formatProbability(result.threshold)}
                      </span>
                    </div>
                  </div>
                  
                  <div className="h-24 w-px bg-slate-700 hidden md:block" />
                  
                  <div className="space-y-2 text-center md:text-right">
                    <h3 className="text-slate-400 font-medium">Prediction</h3>
                    <p className="text-xl font-semibold text-slate-200">
                      {result.prediction === 'flare' ? 'M/X Class Flare' : 'No Major Flare'}
                    </p>
                    <p className="text-sm text-slate-500">Confidence: <span className="capitalize text-slate-300">{result.confidence_level}</span></p>
                  </div>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="md:col-span-1">
                  <CardHeader>
                    <CardTitle>Activation Map</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <GradCAMOverlay 
                      originalSrc={result.heatmap} // mock, assume endpoint returns actual base64
                      heatmapSrc={result.heatmap}
                      overlaySrc={result.overlay}
                      explanationNote={result.explanation_note}
                    />
                  </CardContent>
                </Card>

                <Card className="md:col-span-1">
                  <CardHeader>
                    <CardTitle>Magnetic Features</CardTitle>
                    <CardDescription>Extracted physical properties</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {Object.entries(result.magnetic_features || {}).map(([key, val]) => (
                        <div key={key} className="flex justify-between items-center border-b border-slate-800 pb-2">
                          <span className="text-sm text-slate-400 capitalize">{key.replace(/_/g, ' ')}</span>
                          <span className="font-mono text-sm text-slate-200">{formatMetricValue(val as number)}</span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
