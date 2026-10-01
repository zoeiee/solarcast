'use client';

import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { ConfusionMatrix } from '@/components/charts/ConfusionMatrix';
import { ROCChart } from '@/components/charts/ROCChart';
import { PRChart } from '@/components/charts/PRChart';
import { MetricsTable } from '@/components/charts/MetricsTable';
import { getPerformanceMetrics } from '@/lib/api';
import { PerformanceMetrics as MetricsType } from '@/lib/types';
import { BarChart3, Loader2 } from 'lucide-react';

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<MetricsType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // In production, we'd fetch this from the API.
    // We will provide mock data if the API fails for demo purposes.
    getPerformanceMetrics().then(setMetrics).catch(() => {
      // Fallback mock data
      setMetrics({
        accuracy: 0.912,
        precision: 0.845,
        recall: 0.789,
        f1: 0.816,
        roc_auc: 0.945,
        pr_auc: 0.882,
        tss: 0.712,
        hss: 0.695,
        confusion_matrix: [[1250, 85], [112, 420]], // TN, FP, FN, TP
        roc_curve: Array.from({ length: 20 }, (_, i) => ({ fpr: i/19, tpr: Math.pow(i/19, 0.4) })),
        pr_curve: Array.from({ length: 20 }, (_, i) => ({ recall: i/19, precision: 1 - Math.pow(i/19, 2)*0.4 })),
        is_synthetic: true
      });
    }).finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <Loader2 className="w-8 h-8 animate-spin text-amber-500" />
      </div>
    );
  }

  if (!metrics) return null;

  return (
    <div className="space-y-8 max-w-6xl mx-auto">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
          <BarChart3 className="w-8 h-8 text-amber-500" /> Model Performance
        </h1>
        <p className="text-slate-400">Evaluation metrics on the hold-out test set (partitioned by Active Region).</p>
        {metrics.is_synthetic && (
          <p className="text-amber-500 text-sm mt-2 font-medium bg-amber-500/10 inline-block px-3 py-1 rounded-md">
            Note: Displaying synthetic/mock data for demonstration.
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Confusion Matrix</CardTitle>
            <CardDescription>Threshold = 0.5</CardDescription>
          </CardHeader>
          <CardContent className="flex justify-center p-6">
            <ConfusionMatrix matrix={metrics.confusion_matrix} className="w-full max-w-sm" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Key Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <MetricsTable metrics={metrics} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>ROC Curve</CardTitle>
            <CardDescription>Receiver Operating Characteristic</CardDescription>
          </CardHeader>
          <CardContent>
            <ROCChart data={metrics.roc_curve} auc={metrics.roc_auc} />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Precision-Recall Curve</CardTitle>
            <CardDescription>Handling class imbalance</CardDescription>
          </CardHeader>
          <CardContent>
            <PRChart data={metrics.pr_curve} auc={metrics.pr_auc} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
