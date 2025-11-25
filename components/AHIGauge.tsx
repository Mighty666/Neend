'use client'

import { motion } from 'framer-motion'

interface AHIGaugeProps {
  value: number
  maxValue?: number
  size?: number
}

export function AHIGauge({ value, maxValue = 60, size = 200 }: AHIGaugeProps) {
  const percentage = Math.min((value / maxValue) * 100, 100)
  const circumference = 2 * Math.PI * 45
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  const getColor = () => {
    if (value < 5) return '#4ade80'
    if (value < 15) return '#facc15'
    if (value < 30) return '#fb923c'
    return '#f87171'
  }

  const getSeverity = () => {
    if (value < 5) return 'Normal'
    if (value < 15) return 'Mild'
    if (value < 30) return 'Moderate'
    return 'Severe'
  }

  return (
    <div className="relative" style={{ width: size, height: size }}>
      <svg
        viewBox="0 0 100 100"
        className="transform -rotate-90"
        style={{ width: size, height: size }}
      >
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="rgba(255,255,255,0.1)"
          strokeWidth="8"
        />

        {/* Progress circle */}
        <motion.circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke={getColor()}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, ease: "easeOut" }}
        />
      </svg>

      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="text-3xl font-bold"
          style={{ color: getColor() }}
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
        >
          {value.toFixed(1)}
        </motion.span>
        <span className="text-xs text-gray-400">AHI</span>
        <span className="text-xs mt-1" style={{ color: getColor() }}>
          {getSeverity()}
        </span>
      </div>
    </div>
  )
}
