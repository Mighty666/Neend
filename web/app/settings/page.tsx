'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { BellIcon, ShieldCheckIcon, UserIcon, ArrowRightOnRectangleIcon } from '@heroicons/react/24/outline'
import toast from 'react-hot-toast'

export default function Settings() {
  const [alertThreshold, setAlertThreshold] = useState(30)
  const [notifications, setNotifications] = useState({
    criticalAlerts: true,
    dailySummary: true,
    weeklyReport: false,
    tips: true
  })
  const [privacy, setPrivacy] = useState({
    shareWithClinician: false,
    contributeToResearch: false,
    retentionDays: 30
  })

  const handleSave = () => {
    toast.success('Settings saved')
  }

  return (
    <div className="min-h-screen bg-dark-950">
      <header className="glass-dark sticky top-0 z-50">
        <div className="max-w-2xl mx-auto px-6 py-4 flex justify-between items-center">
          <Link href="/dashboard" className="text-xl font-bold">
            <span className="text-gradient">Neend</span>
            <span className="text-white">AI</span>
          </Link>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-6 py-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-2xl font-bold mb-1">Settings</h1>
          <p className="text-gray-400">Manage your preferences and privacy</p>
        </motion.div>

        {/* Profile Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass rounded-xl p-6 mb-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <UserIcon className="w-5 h-5 text-blue-400" />
            <h2 className="font-semibold">Profile</h2>
          </div>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400">Name</label>
              <input type="text" defaultValue="Maanas" className="w-full bg-dark-900 border border-white/10 rounded-lg py-2 px-3 mt-1" />
            </div>
            <div>
              <label className="text-sm text-gray-400">Email</label>
              <input type="email" defaultValue="maanas@example.com" className="w-full bg-dark-900 border border-white/10 rounded-lg py-2 px-3 mt-1" />
            </div>
            <div>
              <label className="text-sm text-gray-400">Emergency Contact</label>
              <input type="tel" placeholder="+1 (555) 000-0000" className="w-full bg-dark-900 border border-white/10 rounded-lg py-2 px-3 mt-1" />
            </div>
          </div>
        </motion.div>

        {/* Notifications Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass rounded-xl p-6 mb-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <BellIcon className="w-5 h-5 text-yellow-400" />
            <h2 className="font-semibold">Notifications</h2>
          </div>

          <div className="space-y-4">
            <div className="mb-4">
              <label className="text-sm text-gray-400 mb-2 block">Alert Threshold (AHI)</label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="5"
                  max="60"
                  value={alertThreshold}
                  onChange={(e) => setAlertThreshold(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="text-lg font-bold w-12">{alertThreshold}</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">Alert when AHI exceeds this value</p>
            </div>

            {Object.entries(notifications).map(([key, value]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                <button
                  onClick={() => setNotifications({ ...notifications, [key]: !value })}
                  className={`w-12 h-6 rounded-full transition-colors ${value ? 'bg-blue-600' : 'bg-dark-700'}`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transition-transform ${value ? 'translate-x-6' : 'translate-x-0.5'}`} />
                </button>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Privacy Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="glass rounded-xl p-6 mb-6"
        >
          <div className="flex items-center gap-3 mb-4">
            <ShieldCheckIcon className="w-5 h-5 text-green-400" />
            <h2 className="font-semibold">Privacy</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm">Share with Clinician</span>
                <p className="text-xs text-gray-500">Allow your doctor to view reports</p>
              </div>
              <button
                onClick={() => setPrivacy({ ...privacy, shareWithClinician: !privacy.shareWithClinician })}
                className={`w-12 h-6 rounded-full transition-colors ${privacy.shareWithClinician ? 'bg-blue-600' : 'bg-dark-700'}`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${privacy.shareWithClinician ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <span className="text-sm">Contribute to Research</span>
                <p className="text-xs text-gray-500">Anonymized data helps improve accuracy</p>
              </div>
              <button
                onClick={() => setPrivacy({ ...privacy, contributeToResearch: !privacy.contributeToResearch })}
                className={`w-12 h-6 rounded-full transition-colors ${privacy.contributeToResearch ? 'bg-blue-600' : 'bg-dark-700'}`}
              >
                <div className={`w-5 h-5 rounded-full bg-white transition-transform ${privacy.contributeToResearch ? 'translate-x-6' : 'translate-x-0.5'}`} />
              </button>
            </div>

            <div>
              <label className="text-sm text-gray-400 mb-2 block">Data Retention (days)</label>
              <select
                value={privacy.retentionDays}
                onChange={(e) => setPrivacy({ ...privacy, retentionDays: Number(e.target.value) })}
                className="w-full bg-dark-900 border border-white/10 rounded-lg py-2 px-3"
              >
                <option value={7}>7 days</option>
                <option value={30}>30 days</option>
                <option value={90}>90 days</option>
                <option value={365}>1 year</option>
              </select>
            </div>
          </div>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-3"
        >
          <button
            onClick={handleSave}
            className="w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-semibold transition-colors"
          >
            Save Changes
          </button>

          <button className="w-full py-3 glass hover:bg-white/10 rounded-xl font-semibold transition-colors flex items-center justify-center gap-2 text-red-400">
            <ArrowRightOnRectangleIcon className="w-5 h-5" />
            Sign Out
          </button>
        </motion.div>
      </main>
    </div>
  )
}
