import React from 'react';
import { PerformanceMetrics } from '@/lib/types';
import { formatMetricValue } from '@/lib/utils';
import { cn } from '@/lib/utils';

interface MetricsTableProps {
  metrics: PerformanceMetrics;
  className?: string;
}

export function MetricsTable({ metrics, className }: MetricsTableProps) {
  const tableData = [
    { name: 'Accuracy', value: metrics.accuracy, desc: 'Overall correct prediction rate' },
    { name: 'Precision', value: metrics.precision, desc: 'Correct positive predictions / Total positive predictions' },
    { name: 'Recall (TPR)', value: metrics.recall, desc: 'Correct positive predictions / Total actual positives' },
    { name: 'F1 Score', value: metrics.f1, desc: 'Harmonic mean of Precision and Recall' },
    { name: 'ROC AUC', value: metrics.roc_auc, desc: 'Area under the Receiver Operating Characteristic curve' },
    { name: 'PR AUC', value: metrics.pr_auc, desc: 'Area under the Precision-Recall curve' },
    { name: 'True Skill Statistic (TSS)', value: metrics.tss, desc: 'TPR - FPR (Standard for operational space weather)' },
    { name: 'Heidke Skill Score (HSS)', value: metrics.hss, desc: 'Fraction of correct predictions after eliminating random hits' },
  ];

  return (
    <div className={cn("overflow-hidden rounded-xl border border-slate-700 bg-slate-800/50", className)}>
      <table className="w-full text-sm text-left text-slate-300">
        <thead className="text-xs text-slate-400 uppercase bg-slate-900/80 border-b border-slate-700">
          <tr>
            <th scope="col" className="px-6 py-4">Metric</th>
            <th scope="col" className="px-6 py-4">Value</th>
            <th scope="col" className="px-6 py-4">Description</th>
          </tr>
        </thead>
        <tbody>
          {tableData.map((row, idx) => (
            <tr 
              key={row.name} 
              className={cn(
                "border-b border-slate-700 hover:bg-slate-800/80 transition-colors",
                idx % 2 === 0 ? "bg-slate-800/30" : "bg-transparent"
              )}
            >
              <td className="px-6 py-4 font-medium text-slate-100 whitespace-nowrap">
                {row.name}
              </td>
              <td className="px-6 py-4 font-mono text-amber-400">
                {formatMetricValue(row.value)}
              </td>
              <td className="px-6 py-4 text-slate-400">
                {row.desc}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
