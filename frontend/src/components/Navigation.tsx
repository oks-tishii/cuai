import {
  Upload,
  BarChart3,
  History,
  Settings,
  AlertTriangle,
} from "lucide-react";

interface NavigationProps {
  currentScreen: string;
  onScreenChange: (screen: string) => void;
}

export default function Navigation({
  currentScreen,
  onScreenChange,
}: NavigationProps) {
  const navItems = [
    { id: "upload", label: "アップロード", icon: Upload },
    { id: "analysis", label: "詳細分析", icon: BarChart3 },
    { id: "history", label: "履歴管理", icon: History },
    { id: "settings", label: "設定", icon: Settings },
  ];

  return (
    <nav className="bg-gray-800 border-b border-gray-700">
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold text-white">
              PatchCore Anomaly Detection
            </h1>
          </div>

          <div className="flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.id}
                  onClick={() => onScreenChange(item.id)}
                  className={`px-4 py-2 rounded-lg transition-all flex items-center space-x-2 ${
                    currentScreen === item.id
                      ? "bg-blue-600 text-white shadow-lg"
                      : "text-gray-300 hover:bg-gray-700 hover:text-white"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{item.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
