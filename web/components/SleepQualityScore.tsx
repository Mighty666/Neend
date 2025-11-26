'use client'

import { motion } from 'framer-motion'

interface SleepQualityScoreProps {
  score: number
  factors: {
    duration: number
    efficiency: number
    ahi: number
    consistency: number
  }
}

export function SleepQualityScore({ score, factors }: SleepQualityScoreProps) {
  const getScoreColor = () => {
    if (score >= 80) return 'text-green-400'
    if (score >= 60) return 'text-yellow-400'
    if (score >= 40) return 'text-orange-400'
    return 'text-red-400'
  }

  const getScoreLabel = () => {
    if (score >= 80) return 'Excellent'
    if (score >= 60) return 'Good'
    if (score >= 40) return 'Fair'
    return 'Poor'
  }

  const factorLabels = {
    duration: 'Duration',
    efficiency: 'Efficiency',
    ahi: 'Breathing',
    consistency: 'Consistency'
  }

  return (
    <div className="glass rounded-xl p-6">
      <h3 className="text-lg font-semibold mb-4">Sleep Quality Score</h3>

      <div className="flex items-center justify-center mb-6">
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 200, damping: 15 }}
          className="relative w-32 h-32"
        >
          <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
            <circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="rgba(255,255,255,0.1)"
              strokeWidth="8"
            />
            <motion.circle
              cx="50"
              cy="50"
              r="40"
              fill="none"
              stroke="currentColor"
              strokeWidth="8"
              strokeLinecap="round"
              className={getScoreColor()}
              strokeDasharray={251.2}
              initial={{ strokeDashoffset: 251.2 }}
              animate={{ strokeDashoffset: 251.2 - (score / 100) * 251.2 }}
              transition={{ duration: 1, delay: 0.3 }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <motion.span
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className={`text-3xl font-bold ${getScoreColor()}`}
            >
              {score}
            </motion.span>
            <span className="text-xs text-gray-400">{getScoreLabel()}</span>
          </div>
        </motion.div>
      </div>

      {/* Factor Breakdown */}
      <div className="space-y-3">
        {Object.entries(factors).map(([key, value], i) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 + i * 0.1 }}
          >
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">{factorLabels[key as keyof typeof factorLabels]}</span>
              <span>{value}%</span>
            </div>
            <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
              <motion.div
                className={`h-full rounded-full ${value >= 70 ? 'bg-green-500' : value >= 40 ? 'bg-yellow-500' : 'bg-red-500'}`}
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 0.5, delay: 0.8 + i * 0.1 }}
              />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}
