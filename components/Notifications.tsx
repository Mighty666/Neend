'use client'

import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { XMarkIcon, ExclamationTriangleIcon, CheckCircleIcon, InformationCircleIcon } from '@heroicons/react/24/outline'

interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: number
}

interface NotificationsProps {
  notifications: Notification[]
  onDismiss: (id: string) => void
}

export function Notifications({ notifications, onDismiss }: NotificationsProps) {
  const icons = {
    success: <CheckCircleIcon className="w-5 h-5 text-green-400" />,
    error: <XMarkIcon className="w-5 h-5 text-red-400" />,
    warning: <ExclamationTriangleIcon className="w-5 h-5 text-yellow-400" />,
    info: <InformationCircleIcon className="w-5 h-5 text-blue-400" />,
  }

  const colors = {
    success: 'border-green-500/30 bg-green-500/10',
    error: 'border-red-500/30 bg-red-500/10',
    warning: 'border-yellow-500/30 bg-yellow-500/10',
    info: 'border-blue-500/30 bg-blue-500/10',
  }

  return (
    <div className="fixed top-20 right-6 z-50 space-y-3 max-w-sm">
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, x: 100, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 100, scale: 0.9 }}
            className={`p-4 rounded-xl border backdrop-blur-lg ${colors[notification.type]}`}
          >
            <div className="flex items-start gap-3">
              {icons[notification.type]}
              <div className="flex-1">
                <p className="font-medium text-sm">{notification.title}</p>
                <p className="text-xs text-gray-400 mt-1">{notification.message}</p>
              </div>
              <button
                onClick={() => onDismiss(notification.id)}
                className="text-gray-400 hover:text-white"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

// Hook for managing notifications
export function useNotifications() {
  const [notifications, setNotifications] = useState<Notification[]>([])

  const addNotification = (
    type: Notification['type'],
    title: string,
    message: string,
    duration: number = 5000
  ) => {
    const id = Math.random().toString(36).substr(2, 9)
    const notification: Notification = {
      id,
      type,
      title,
      message,
      timestamp: Date.now(),
    }

    setNotifications((prev) => [...prev, notification])

    if (duration > 0) {
      setTimeout(() => {
        dismissNotification(id)
      }, duration)
    }

    return id
  }

  const dismissNotification = (id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id))
  }

  return {
    notifications,
    addNotification,
    dismissNotification,
    success: (title: string, message: string) => addNotification('success', title, message),
    error: (title: string, message: string) => addNotification('error', title, message),
    warning: (title: string, message: string) => addNotification('warning', title, message),
    info: (title: string, message: string) => addNotification('info', title, message),
  }
}
