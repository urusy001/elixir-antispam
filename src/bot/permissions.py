from aiogram.types import ChatPermissions

NEW_USER = ChatPermissions(
    can_send_messages=False,
    can_send_audios=False,
    can_send_documents=False,
    can_send_photos=False,
    can_send_videos=False,
    can_send_video_notes=False,
    can_send_polls=False,
    can_send_other_messages=False,
    can_add_web_page_previews=False,
    can_change_info=False,
    can_invite_users=False,
    can_pin_messages=False,
    can_manage_topics=False,
)

USER_PASSED = ChatPermissions(
    can_send_messages=True,
    can_send_audios=True,
    can_send_photos=True,
    can_send_videos=True,
    can_send_video_notes=True,
    can_send_polls=True,
    can_send_other_messages=True,
)