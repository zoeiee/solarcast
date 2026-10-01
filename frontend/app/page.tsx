import Link from 'next/link';
import { ArrowRight, Activity, Brain, BarChart3, Database } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';

export default function HomePage() {
  return (
    <div className="flex flex-col gap-12 pb-12">
      {/* Hero Section */}
      <section className="text-center py-16 md:py-24 space-y-6">
        <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-600">
          SolarCast
        </h1>
        <p className="text-xl md:text-2xl text-slate-300 max-w-3xl mx-auto font-light">
          Machine-learning analysis of solar active regions for flare-risk estimation.
        </p>
        <div className="flex justify-center pt-4">
          <Link href="/analyze">
            <Button size="lg" className="gap-2 text-lg">
              Analyze a Magnetogram <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Pipeline Diagram */}
      <section className="max-w-4xl mx-auto w-full">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4 p-8 bg-slate-900/50 rounded-2xl border border-slate-800">
          <div className="flex flex-col items-center text-center space-y-2">
            <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center mb-2 shadow-lg">
              <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-blue-900 to-slate-400 animate-pulse-slow" />
            </div>
            <span className="font-semibold text-slate-200">Magnetogram</span>
          </div>
          
          <ArrowRight className="w-6 h-6 text-slate-600 hidden md:block" />
          <div className="w-1 h-6 bg-slate-700 md:hidden" />

          <div className="flex flex-col items-center text-center space-y-2">
            <div className="w-16 h-16 rounded-xl bg-amber-500/20 border border-amber-500/30 flex items-center justify-center mb-2 text-amber-500">
              <Brain className="w-8 h-8" />
            </div>
            <span className="font-semibold text-slate-200">CNN Model</span>
          </div>

          <ArrowRight className="w-6 h-6 text-slate-600 hidden md:block" />
          <div className="w-1 h-6 bg-slate-700 md:hidden" />

          <div className="flex flex-col items-center text-center space-y-2">
            <div className="w-16 h-16 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center mb-2">
              <span className="text-xl font-bold text-red-400">72%</span>
            </div>
            <span className="font-semibold text-slate-200">Flare Risk</span>
          </div>

          <ArrowRight className="w-6 h-6 text-slate-600 hidden md:block" />
          <div className="w-1 h-6 bg-slate-700 md:hidden" />

          <div className="flex flex-col items-center text-center space-y-2">
            <div className="w-16 h-16 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center mb-2 text-emerald-400">
              <Activity className="w-8 h-8" />
            </div>
            <span className="font-semibold text-slate-200">Explanation</span>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-5xl mx-auto w-full">
        <Card>
          <CardHeader>
            <Brain className="w-8 h-8 text-amber-500 mb-2" />
            <CardTitle>Deep Learning Analysis</CardTitle>
            <CardDescription>CNN-based active region classification</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300">
              SolarCast utilizes a custom Convolutional Neural Network trained on years of SDO/HMI magnetograms to identify complex magnetic topologies associated with major solar flares.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Activity className="w-8 h-8 text-blue-400 mb-2" />
            <CardTitle>Explainable AI</CardTitle>
            <CardDescription>Grad-CAM activation mapping</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300">
              Understand model decisions with visual heatmaps highlighting the exact magnetic features—like high-gradient polarity inversion lines—that drive the flare prediction.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <BarChart3 className="w-8 h-8 text-emerald-400 mb-2" />
            <CardTitle>Space Weather Metrics</CardTitle>
            <CardDescription>TSS, HSS, and rigorous evaluation</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300">
              Evaluated using standard operational space weather metrics, including True Skill Statistic (TSS) and Heidke Skill Score (HSS), ensuring practical reliability.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Database className="w-8 h-8 text-purple-400 mb-2" />
            <CardTitle>Scientific Rigor</CardTitle>
            <CardDescription>Active-region-aware data handling</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-slate-300">
              Training and testing data splits are strictly partitioned by NOAA Active Region (AR) numbers to prevent data leakage and ensure robust generalization.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
