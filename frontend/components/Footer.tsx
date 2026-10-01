export default function Footer() {
  return (
    <footer className="bg-slate-950 border-t border-slate-800 mt-auto">
      <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="text-center md:text-left">
            <p className="text-slate-100 font-medium">SolarCast &mdash; Solar Flare Prediction</p>
            <p className="text-sm text-slate-400 mt-1">Research project. Not for operational space weather forecasting.</p>
          </div>
          <div className="text-sm text-slate-500 text-center md:text-right">
            <p>Built with PyTorch, FastAPI, Next.js</p>
            <a href="#" className="text-amber-500 hover:text-amber-400 mt-1 inline-block">GitHub Repository</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
