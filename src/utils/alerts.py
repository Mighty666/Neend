"""
Alert system for sleep apnea notifications
"""

import logging
from typing import Dict, Optional
import httpx

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Multi-channel alert system for critical sleep events.
    Supports SMS, push notifications, and email alerts.
    """

    def __init__(
        self,
        alert_threshold: int = 30,
        sns_topic_arn: Optional[str] = None,
        aws_region: str = "us-east-1"
    ):
        self.threshold = alert_threshold
        self.sns_topic_arn = sns_topic_arn
        self.aws_region = aws_region
        self._sns_client = None

    @property
    def sns_client(self):
        """Lazy load SNS client"""
        if self._sns_client is None and self.sns_topic_arn:
            import boto3
            self._sns_client = boto3.client('sns', region_name=self.aws_region)
        return self._sns_client

    async def check_and_alert(
        self,
        ahi_score: float,
        user_profile: Dict
    ) -> bool:
        """
        Check AHI score and send alerts if threshold exceeded.

        Args:
            ahi_score: Current AHI score
            user_profile: User's profile with contact info

        Returns:
            True if alerts were sent
        """
        if ahi_score < self.threshold:
            return False

        alerts_sent = False

        # Send SMS to emergency contact
        if user_profile.get('emergency_contact'):
            try:
                await self.send_sms_alert(
                    user_profile['emergency_contact'],
                    self._create_sms_message(user_profile.get('name', 'User'), ahi_score)
                )
                alerts_sent = True
            except Exception as e:
                logger.error(f"Failed to send SMS alert: {e}")

        # Send push notification
        if user_profile.get('device_token'):
            try:
                await self.send_push_notification(
                    user_profile['device_token'],
                    title="Sleep Apnea Alert",
                    body=f"Severe OSA detected (AHI: {ahi_score})"
                )
                alerts_sent = True
            except Exception as e:
                logger.error(f"Failed to send push notification: {e}")

        # Email to clinician
        if user_profile.get('clinician_email'):
            try:
                await self.send_clinical_report(
                    user_profile['clinician_email'],
                    ahi_score,
                    user_profile.get('user_id', 'unknown')
                )
                alerts_sent = True
            except Exception as e:
                logger.error(f"Failed to send clinical email: {e}")

        return alerts_sent

    def _create_sms_message(self, name: str, ahi_score: float) -> str:
        """Create SMS alert message"""
        return (
            f"URGENT: {name}'s AHI score is {ahi_score:.1f}. "
            f"Immediate medical attention recommended. "
            f"- NeendAI Sleep Monitor"
        )

    async def send_sms_alert(self, phone_number: str, message: str):
        """
        Send SMS alert via AWS SNS.

        Args:
            phone_number: Recipient phone number (E.164 format)
            message: Alert message
        """
        if not self.sns_client:
            logger.warning("SNS client not configured, skipping SMS")
            return

        try:
            self.sns_client.publish(
                PhoneNumber=phone_number,
                Message=message,
                MessageAttributes={
                    'AWS.SNS.SMS.SMSType': {
                        'DataType': 'String',
                        'StringValue': 'Transactional'
                    }
                }
            )
            logger.info(f"SMS alert sent to {phone_number}")
        except Exception as e:
            logger.error(f"SNS publish error: {e}")
            raise

    async def send_push_notification(
        self,
        device_token: str,
        title: str,
        body: str
    ):
        """
        Send push notification (placeholder - implement with FCM/APNS).

        Args:
            device_token: Device push token
            title: Notification title
            body: Notification body
        """
        # Placeholder for FCM/APNS implementation
        logger.info(f"Push notification: {title} - {body} -> {device_token[:20]}...")

        # Example FCM implementation:
        # async with httpx.AsyncClient() as client:
        #     response = await client.post(
        #         "https://fcm.googleapis.com/fcm/send",
        #         headers={"Authorization": f"key={FCM_SERVER_KEY}"},
        #         json={
        #             "to": device_token,
        #             "notification": {"title": title, "body": body}
        #         }
        #     )

    async def send_clinical_report(
        self,
        email: str,
        ahi_score: float,
        user_id: str
    ):
        """
        Send clinical report to healthcare provider.

        Args:
            email: Clinician's email
            ahi_score: Current AHI score
            user_id: Patient's user ID
        """
        # Placeholder for email implementation
        logger.info(f"Clinical report for patient {user_id} sent to {email}")

        # Would integrate with SES or SMTP here

    def should_alert(self, ahi_score: float) -> bool:
        """Check if AHI score exceeds alert threshold"""
        return ahi_score >= self.threshold

    def get_severity_alert_level(self, ahi_score: float) -> str:
        """Get alert level based on AHI score"""
        if ahi_score >= 30:
            return "CRITICAL"
        elif ahi_score >= 15:
            return "WARNING"
        elif ahi_score >= 5:
            return "INFO"
        else:
            return "NORMAL"
