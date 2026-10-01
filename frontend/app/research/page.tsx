import React from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Library } from 'lucide-react';

export default function ResearchPage() {
  const papers = [
    {
      title: "Deep Learning for Solar Flare Forecasting: Current Status and Future Prospects",
      authors: "Camporeale, E., et al.",
      year: 2019,
      journal: "Space Weather",
      description: "A comprehensive review of ML techniques applied to space weather, outlining the challenges of class imbalance and active-region splitting.",
      category: "Flare Prediction"
    },
    {
      title: "Predicting Solar Flares Using a Long Short-term Memory Network",
      authors: "Liu, H., et al.",
      year: 2019,
      journal: "The Astrophysical Journal",
      description: "Demonstrates the use of time-series SHARP parameters for flare prediction, establishing baseline metrics (TSS, HSS) for the field.",
      category: "Flare Prediction"
    },
    {
      title: "The Helioseismic and Magnetic Imager (HMI) Investigation for the Solar Dynamics Observatory (SDO)",
      authors: "Scherrer, P. H., et al.",
      year: 2012,
      journal: "Solar Physics",
      description: "The foundational paper detailing the HMI instrument which provides the vector magnetogram data used in this project.",
      category: "Data Sources"
    },
    {
      title: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
      authors: "Selvaraju, R. R., et al.",
      year: 2017,
      journal: "ICCV",
      description: "The original paper for the Grad-CAM technique used in our Explainable AI pipeline.",
      category: "Explainability"
    }
  ];

  const grouped = papers.reduce((acc, paper) => {
    if (!acc[paper.category]) acc[paper.category] = [];
    acc[paper.category].push(paper);
    return acc;
  }, {} as Record<string, typeof papers>);

  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-12">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
          <Library className="w-8 h-8 text-amber-500" /> Research & References
        </h1>
        <p className="text-slate-400">Key literature and data sources influencing the development of SolarCast.</p>
      </div>

      <div className="space-y-8">
        {Object.entries(grouped).map(([category, items]) => (
          <div key={category} className="space-y-4">
            <h2 className="text-xl font-semibold text-amber-500 border-b border-slate-800 pb-2">{category}</h2>
            <div className="grid grid-cols-1 gap-4">
              {items.map((paper, idx) => (
                <Card key={idx} className="bg-slate-900/50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-slate-200">{paper.title}</CardTitle>
                    <CardDescription className="text-slate-400">
                      {paper.authors} ({paper.year}) — <span className="italic text-slate-300">{paper.journal}</span>
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-slate-300">{paper.description}</p>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
