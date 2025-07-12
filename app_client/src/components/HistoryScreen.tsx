import React, { useState } from 'react';
import { Search, Filter, Calendar, Download, Eye, Trash2 } from 'lucide-react';

interface DetectionResult {
  id: string;
  image: string;
  anomalyScore: number;
  isAnomalous: boolean;
  heatmap: string;
  markedImage: string;
  timestamp: Date;
}

interface HistoryScreenProps {
  history: DetectionResult[];
  onSelectResult: (result: DetectionResult) => void;
  onDeleteResult: (id: string) => void;
}

export default function HistoryScreen({ history, onSelectResult, onDeleteResult }: HistoryScreenProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'normal' | 'anomaly'>('all');
  const [sortBy, setSortBy] = useState<'date' | 'score'>('date');

  const filteredHistory = history
    .filter(result => {
      if (filterType === 'normal') return !result.isAnomalous;
      if (filterType === 'anomaly') return result.isAnomalous;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'date') {
        return b.timestamp.getTime() - a.timestamp.getTime();
      } else {
        return b.anomalyScore - a.anomalyScore;
      }
    });

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">履歴管理</h2>
          <p className="text-gray-400">過去の検知結果の確認と管理</p>
        </div>

        {/* Controls */}
        <div className="bg-gray-800 rounded-xl p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="検索..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Filter */}
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">すべて</option>
              <option value="normal">正常のみ</option>
              <option value="anomaly">異常のみ</option>
            </select>

            {/* Sort */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="date">日時順</option>
              <option value="score">スコア順</option>
            </select>

            {/* Export */}
            <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center justify-center space-x-2">
              <Download className="w-4 h-4" />
              <span>一括エクスポート</span>
            </button>
          </div>
        </div>

        {/* Results Grid */}
        {filteredHistory.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredHistory.map((result) => (
              <div key={result.id} className="bg-gray-800 rounded-xl overflow-hidden hover:bg-gray-750 transition-colors">
                <div className="aspect-video bg-gray-700 relative overflow-hidden">
                  <img
                    src={result.image}
                    alt="Detection result"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${
                        result.isAnomalous
                          ? 'bg-red-500 text-white'
                          : 'bg-green-500 text-white'
                      }`}
                    >
                      {result.isAnomalous ? '異常' : '正常'}
                    </span>
                  </div>
                </div>

                <div className="p-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-gray-400">
                      {result.timestamp.toLocaleDateString()} {result.timestamp.toLocaleTimeString()}
                    </span>
                  </div>

                  <div className="mb-3">
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm text-gray-400">異常度</span>
                      <span className="text-sm font-mono">
                        {(result.anomalyScore * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          result.isAnomalous
                            ? 'bg-gradient-to-r from-red-500 to-red-600'
                            : 'bg-gradient-to-r from-green-500 to-green-600'
                        }`}
                        style={{ width: `${result.anomalyScore * 100}%` }}
                      ></div>
                    </div>
                  </div>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => onSelectResult(result)}
                      className="flex-1 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors flex items-center justify-center space-x-1"
                    >
                      <Eye className="w-4 h-4" />
                      <span className="text-sm">詳細</span>
                    </button>
                    <button
                      onClick={() => onDeleteResult(result.id)}
                      className="p-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="bg-gray-800 rounded-xl p-12 text-center">
            <Calendar className="w-20 h-20 text-gray-500 mx-auto mb-6" />
            <h3 className="text-xl font-semibold mb-2">履歴がありません</h3>
            <p className="text-gray-400">検知結果がここに表示されます</p>
          </div>
        )}

        {/* Summary Stats */}
        {history.length > 0 && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="bg-gray-800 rounded-xl p-6 text-center">
              <div className="text-2xl font-bold text-blue-400">{history.length}</div>
              <div className="text-sm text-gray-400">総検知数</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 text-center">
              <div className="text-2xl font-bold text-green-400">
                {history.filter(r => !r.isAnomalous).length}
              </div>
              <div className="text-sm text-gray-400">正常</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 text-center">
              <div className="text-2xl font-bold text-red-400">
                {history.filter(r => r.isAnomalous).length}
              </div>
              <div className="text-sm text-gray-400">異常</div>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 text-center">
              <div className="text-2xl font-bold text-yellow-400">
                {history.length > 0 ? ((history.filter(r => r.isAnomalous).length / history.length) * 100).toFixed(1) : 0}%
              </div>
              <div className="text-sm text-gray-400">異常率</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}