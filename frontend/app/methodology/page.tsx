import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { BookOpen } from 'lucide-react';

export default function MethodologyPage() {
  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-12">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-100 flex items-center gap-2">
          <BookOpen className="w-8 h-8 text-amber-500" /> Methodology
        </h1>
        <p className="text-slate-400">Technical details on dataset, architecture, and evaluation strategy.</p>
      </div>

      <div className="space-y-6 text-slate-300 leading-relaxed">
        <Card>
          <CardHeader>
            <CardTitle>1. Dataset Construction</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              The models are trained using Space-weather HMI Active Region Patches (SHARP) from the SDO/HMI instrument. Specifically, we use the radial component of the magnetic field (Br) derived from vector magnetograms.
            </p>
            <ul className="list-disc pl-6 space-y-2 text-slate-400">
              <li><strong>Source:</strong> SDO/HMI via JSOC</li>
              <li><strong>Labeling:</strong> GOES X-ray flare catalog</li>
              <li><strong>Window:</strong> Predicting M- or X-class flares occurring within 24 hours of the observation time.</li>
            </ul>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>2. Preprocessing & Augmentation</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Magnetograms contain extreme values (often &gt; 3000 Gauss) but the bulk of the field is weak. To normalize:
            </p>
            <pre className="p-4 bg-slate-900 rounded-md overflow-x-auto text-sm border border-slate-700">
              <code>
                {`# Pseudo-code for preprocessing
img = clip(img, min=-2000, max=2000)
img = img / 2000.0  # Scale to [-1, 1]
img = resize(img, (256, 256))`}
              </code>
            </pre>
            <p>
              During training, we apply random rotations, flips, and minor cropping to ensure the model learns rotation-invariant magnetic features.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>3. Active-Region Aware Splitting</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              A critical pitfall in solar flare prediction is data leakage across time. An active region persists for days. If images of the same active region appear in both the training and test sets, the evaluation will be overly optimistic.
            </p>
            <p>
              <strong>Solution:</strong> We partition the dataset strictly by NOAA Active Region (AR) number. ARs in the test set never appear in the training set.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>4. Architecture & Loss</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              We utilize a custom Convolutional Neural Network (CNN) optimized for spatial feature extraction in scientific imagery.
            </p>
            <p>
              Due to severe class imbalance (M/X flares are rare compared to quiet periods or C-class flares), we employ a Weighted Binary Cross-Entropy (BCE) Loss, applying a higher penalty for false negatives.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>5. Explainability (Grad-CAM)</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Gradient-weighted Class Activation Mapping (Grad-CAM) is used to produce coarse localization maps highlighting the important regions in the image for predicting the concept.
            </p>
            <p>
              By computing the gradient of the flare probability with respect to the final convolutional feature maps, we can overlay a heatmap on the original magnetogram. This often highlights Polarity Inversion Lines (PILs) with strong gradients, aligning with domain knowledge of solar physics.
            </p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
