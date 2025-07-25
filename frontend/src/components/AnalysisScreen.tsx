import { useState, useEffect } from "react";
import {
  Eye,
  Layers,
  Download,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Move,
  List,
} from "lucide-react";

interface DetectionResult {
  id: string;
  image: string;
  anomalyScore: number;
  isAnomalous: boolean;
  heatmap: string;
  markedImage: string;
  timestamp: Date;
  processingTime: number;
  confidence: number;
  numAnomalyRegions: number;
  maxAnomalyScore: number;
}

interface AnalysisScreenProps {
  detectionResults: DetectionResult[];
  threshold: number;
}

export default function AnalysisScreen({
  detectionResults,
  threshold,
}: AnalysisScreenProps) {
  const [selectedResult, setSelectedResult] = useState<DetectionResult | null>(
    null
  );
  const [viewMode, setViewMode] = useState<"original" | "heatmap" | "marked">(
    "original"
  );
  const [zoom, setZoom] = useState(100);

  useEffect(() => {
    if (detectionResults.length > 0 && !selectedResult) {
      setSelectedResult(detectionResults[0]);
    }
  }, [detectionResults, selectedResult]);

  if (detectionResults.length === 0 || !selectedResult) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-6">
        <div className="max-w-6xl mx-auto">
          <div className="mb-8">
            <h2 className="text-3xl font-bold mb-2">詳細分析</h2>
            <p className="text-gray-400">検知結果の詳細な分析と可視化</p>
          </div>

          <div className="bg-gray-800 rounded-xl p-12 text-center">
            <Eye className="w-20 h-20 text-gray-500 mx-auto mb-6" />
            <h3 className="text-xl font-semibold mb-2">分析結果がありません</h3>
            <p className="text-gray-400">
              まず画像をアップロードして異常検知を実行してください
            </p>
          </div>
        </div>
      </div>
    );
  }

  const viewModes = [
    { id: "original", label: "元画像", icon: Eye },
    { id: "heatmap", label: "ヒートマップ", icon: Layers },
    { id: "marked", label: "マーキング画像", icon: Move },
  ];

  const getCurrentImage = () => {
    if (!selectedResult) return "";
    switch (viewMode) {
      case "heatmap":
        return selectedResult.heatmap;
      case "marked":
        return selectedResult.markedImage;
      default:
        return selectedResult.image;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">詳細分析</h2>
          <p className="text-gray-400">
            {detectionResults.length} 件の画像の結果を表示中
          </p>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          {/* Results List */}
          <div className="xl:col-span-1 space-y-4 bg-gray-800 rounded-xl p-4 max-h-[80vh] overflow-y-auto">
            <h3 className="text-lg font-semibold mb-2 flex items-center"><List className="w-5 h-5 mr-2"/>検出リスト</h3>
            {detectionResults.map((result) => (
              <div
                key={result.id}
                onClick={() => setSelectedResult(result)}
                className={`p-3 rounded-lg cursor-pointer transition-colors ${
                  selectedResult?.id === result.id
                    ? "bg-blue-600"
                    : "bg-gray-700 hover:bg-gray-600"
                }`}>
                <div className="flex items-center space-x-4">
                  <img src={result.image} alt="thumbnail" className="w-16 h-16 object-cover rounded-md"/>
                  <div>
                    <div className={`font-semibold ${
                      result.isAnomalous ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {result.isAnomalous ? "異常検出" : "正常"}
                    </div>
                    <div className="text-sm text-gray-400">
                      Score: {(result.anomalyScore * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Main Image Display */}
          <div className="xl:col-span-2 space-y-6">
            {/* View Mode Selector */}
            <div className="bg-gray-800 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">表示モード</h3>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setZoom(Math.max(25, zoom - 25))}
                    className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    <ZoomOut className="w-4 h-4" />
                  </button>
                  <span className="text-sm font-mono w-12 text-center">
                    {zoom}%
                  </span>
                  <button
                    onClick={() => setZoom(Math.min(200, zoom + 25))}
                    className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => setZoom(100)}
                    className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex space-x-2">
                {viewModes.map((mode) => {
                  const Icon = mode.icon;
                  return (
                    <button
                      key={mode.id}
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      onClick={() => setViewMode(mode.id as any)}
                      className={`px-4 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                        viewMode === mode.id
                          ? "bg-blue-600 text-white"
                          : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      <span>{mode.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Image Display */}
            <div className="bg-gray-800 rounded-xl p-6">
              <div className="relative overflow-auto max-h-[600px] bg-gray-900 rounded-lg">
                <img
                  src={getCurrentImage()}
                  alt={`${viewMode} view`}
                  className="transition-all duration-300"
                  style={{
                    transform: `scale(${zoom / 100})`,
                    transformOrigin: "top left",
                  }}
                />
              </div>

              {viewMode === "heatmap" && (
                <div className="mt-4 p-3 bg-gray-700 rounded-lg">
                  <div className="flex items-center justify-between text-sm">
                    <span>異常度の色分け:</span>
                    <div className="flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 bg-blue-500 rounded"></div>
                        <span>正常</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 bg-yellow-500 rounded"></div>
                        <span>注意</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 bg-red-500 rounded"></div>
                        <span>異常</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Panel */}
          <div className="space-y-6">
            {/* Anomaly Score */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">異常度スコア</h3>

              <div className="space-y-4">
                <div className="text-center">
                  <div className="text-3xl font-bold mb-2">
                    {(selectedResult.anomalyScore * 100).toFixed(1)}%
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-medium ${
                      selectedResult.isAnomalous
                        ? "bg-red-500 text-white"
                        : "bg-green-500 text-white"
                    }`}
                  >
                    {selectedResult.isAnomalous ? "異常検出" : "正常"}
                  </div>
                </div>

                <div className="relative">
                  <div className="w-full bg-gray-700 rounded-full h-4">
                    <div
                      className={`h-4 rounded-full transition-all duration-1000 ${
                        selectedResult.anomalyScore > threshold
                          ? "bg-gradient-to-r from-red-500 to-red-600"
                          : "bg-gradient-to-r from-green-500 to-green-600"
                      }`}
                      style={{
                        width: `${selectedResult.anomalyScore * 100}%`,
                      }}
                    ></div>
                  </div>
                  <div
                    className="absolute top-0 w-0.5 h-4 bg-yellow-400"
                    style={{ left: `${threshold * 100}%` }}
                  ></div>
                </div>

                <div className="text-xs text-gray-400 text-center">
                  しきい値: {(threshold * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* Statistics */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">統計情報</h3>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-400">処理日時</span>
                  <span className="font-mono">{selectedResult.timestamp.toLocaleTimeString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">処理時間</span>
                  <span className="font-mono">{selectedResult.processingTime.toFixed(3)}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">信頼度</span>
                  <span className="font-mono">{(selectedResult.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">検出領域数</span>
                  <span className="font-mono">
                    {selectedResult.numAnomalyRegions}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">最大異常度</span>
                  <span className="font-mono">
                    {(selectedResult.maxAnomalyScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Export Options */}
            <div className="bg-gray-800 rounded-xl p-6">
              <h3 className="text-lg font-semibold mb-4">エクスポート</h3>

              <div className="space-y-3">
                <button className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center justify-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>選択中の結果を保存</span>
                </button>
                <button className="w-full py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors flex items-center justify-center space-x-2">
                  <Download className="w-4 h-4" />
                  <span>すべての結果を保存 (ZIP)</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
