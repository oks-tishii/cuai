import { Sliders, Database, Bell, Shield } from "lucide-react";

interface SettingsScreenProps {
  threshold: number;
  onThresholdChange: (value: number) => void;
}

export default function SettingsScreen({
  threshold,
  onThresholdChange,
}: SettingsScreenProps) {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h2 className="text-3xl font-bold mb-2">設定</h2>
          <p className="text-gray-400">PatchCore異常検知の詳細設定</p>
        </div>

        <div className="space-y-6">
          {/* Detection Settings */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Sliders className="w-6 h-6 text-blue-400" />
              <h3 className="text-xl font-semibold">検知設定</h3>
            </div>

            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium mb-3">
                  異常度しきい値
                </label>
                <div className="space-y-3">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={threshold}
                    onChange={(e) =>
                      onThresholdChange(parseFloat(e.target.value))
                    }
                    className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-sm text-gray-400">
                    <span>0.0 (低感度)</span>
                    <span className="font-mono text-white bg-gray-700 px-2 py-1 rounded">
                      {threshold.toFixed(2)}
                    </span>
                    <span>1.0 (高感度)</span>
                  </div>
                  <p className="text-sm text-gray-500">
                    しきい値を下げると異常検知の感度が上がり、より多くの異常を検出します
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    パッチサイズ
                  </label>
                  <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <option value="3">3x3</option>
                    <option value="5">5x5</option>
                    <option value="7">7x7</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    特徴抽出レイヤー
                  </label>
                  <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <option value="layer2">Layer 2</option>
                    <option value="layer3">Layer 3</option>
                    <option value="layer4">Layer 4</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Model Settings */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Database className="w-6 h-6 text-green-400" />
              <h3 className="text-xl font-semibold">モデル設定</h3>
            </div>

            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    バックボーンモデル
                  </label>
                  <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <option value="resnet18">ResNet-18</option>
                    <option value="resnet50">ResNet-50</option>
                    <option value="efficientnet">EfficientNet</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">
                    メモリバンクサイズ
                  </label>
                  <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500">
                    <option value="1000">1,000</option>
                    <option value="5000">5,000</option>
                    <option value="10000">10,000</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  />
                  <span className="text-sm">GPU加速を使用する</span>
                </label>
              </div>
            </div>
          </div>

          {/* Notification Settings */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Bell className="w-6 h-6 text-yellow-400" />
              <h3 className="text-xl font-semibold">通知設定</h3>
            </div>

            <div className="space-y-4">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  defaultChecked
                />
                <span className="text-sm">異常検出時に通知する</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm">処理完了時に音で知らせる</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm">メール通知を有効にする</span>
              </label>
            </div>
          </div>

          {/* Security Settings */}
          <div className="bg-gray-800 rounded-xl p-6">
            <div className="flex items-center space-x-3 mb-6">
              <Shield className="w-6 h-6 text-purple-400" />
              <h3 className="text-xl font-semibold">セキュリティ設定</h3>
            </div>

            <div className="space-y-4">
              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  defaultChecked
                />
                <span className="text-sm">
                  アップロード画像を自動削除する (24時間後)
                </span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm">検知結果をローカルに保存する</span>
              </label>

              <label className="flex items-center space-x-3">
                <input
                  type="checkbox"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                />
                <span className="text-sm">匿名使用統計を送信する</span>
              </label>
            </div>
          </div>

          {/* Save Button */}
          <div className="flex justify-end">
            <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors font-medium">
              設定を保存
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
