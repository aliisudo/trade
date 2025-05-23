#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import time
import threading
import requests
import logging
import asyncio
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)
# غیرفعال کردن هشدار SSL نامعتبر (با احتیاط)
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# تنظیمات
BOT_TOKEN = "6171020622:AAGPztIwjw4Q_I4vEIw3_zLyKw75HzWs2-o"
ADMIN_CHAT_ID = 1069585446
CHANNEL_ID = "@solmimcoins"
DEFAULT_BACKUP_INTERVAL = 30  # دقیقه
MIN_BACKUP_INTERVAL = 10 # دقیقه
# --- پایان تنظیمات ---

# تنظیمات لاگ
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("xui_backup.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# حالت‌های مکالمه
PANEL_URL, USERNAME, PASSWORD, CONFIRMATION = range(4)
SET_BACKUP_INTERVAL = 4
CONFIRM_DELETE = 5 # <--- حالت جدید برای تایید حذف
ADD_ADMIN, REMOVE_ADMIN = range(7, 9)  # اضافه کردن states جدید

class XUIBackupBot:
    def __init__(self):
        self.db_conn = sqlite3.connect("xui_panels.db", check_same_thread=False)
        self.OWNER_ID = 1069585446  # آیدی عددی شما به عنوان صاحب اصلی ربات
        self._init_db()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "*/*",
            }
        )
        self.session.verify = False # غیرفعال کردن بررسی SSL
        self.backup_lock = threading.Lock()
        self.backup_interval_seconds = int(self._get_setting("backup_interval", DEFAULT_BACKUP_INTERVAL)) * 60

    def _init_db(self):
        with self.db_conn:
            # جدول‌های قبلی
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS panels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    panel_url TEXT UNIQUE,
                    username TEXT,
                    password TEXT,
                    last_backup TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # جدول جدید برای ادمین‌ها
            self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    user_id INTEGER PRIMARY KEY,
                    added_by INTEGER,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (added_by) REFERENCES admins (user_id)
                )
            """)
            
            # اضافه کردن صاحب ربات به عنوان اولین ادمین اگر در جدول نباشد
            self.db_conn.execute("""
                INSERT OR IGNORE INTO admins (user_id, added_by)
                VALUES (?, ?)
            """, (self.OWNER_ID, self.OWNER_ID))

    def _get_setting(self, key, default=None):
        # ... (بدون تغییر) ...
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        result = cursor.fetchone()
        return result[0] if result else default

    def _set_setting(self, key, value):
        # ... (بدون تغییر) ...
        with self.db_conn:
            self.db_conn.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value)),
            )

    async def get_backup(self, panel_url, username, password):
        # ... (بدون تغییر) ...
        try:
            panel_url = panel_url.rstrip("/")
            login_url = f"{panel_url}/login"
            login_data = {"username": username, "password": password}

            logger.info(f"Attempting login to {panel_url}")
            response = self.session.post(
                login_url, data=login_data, timeout=20, allow_redirects=True
            )
            response.raise_for_status()
            logger.info(f"Login successful for {panel_url}")

            backup_urls = [
                f"{panel_url}/server/getDb",
                f"{panel_url}/panel/api/inbounds/getBackups",
                f"{panel_url}/api/getBackup"
            ]

            for backup_url in backup_urls:
                try:
                    logger.info(f"Attempting backup from {backup_url}")
                    backup_response = self.session.get(backup_url, timeout=30)
                    backup_response.raise_for_status()
                    if backup_response.content:
                        logger.info(f"Backup successful from {backup_url}")
                        return backup_response.content
                    else:
                        logger.warning(f"Empty backup content from {backup_url}")
                except requests.exceptions.RequestException as req_err:
                    logger.warning(f"Failed to get backup from {backup_url}: {req_err}")
                    continue

            raise Exception("هیچ یک از endpointهای بکاپ شناخته شده کار نکرد یا بکاپ خالی بود.")

        except requests.exceptions.Timeout:
             raise Exception(f"خطا در ارتباط با پنل (Timeout): {panel_url}")
        except requests.exceptions.RequestException as e:
             raise Exception(f"خطا در ارتباط با پنل: {str(e)}")
        except Exception as e:
            logger.exception(f"Unexpected error during get_backup for {panel_url}")
            raise Exception(f"خطا در دریافت بکاپ: {str(e)}")


    async def backup_single_panel(
        self, panel_id, panel_url, username, password, context
    ):
        # ... (بدون تغییر) ...
        try:
            logger.info(f"Starting backup for panel ID {panel_id} ({panel_url})")
            backup_data = await self.get_backup(panel_url, username, password)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            panel_name_part = panel_url.replace("http://", "").replace("https://", "").replace(":", "_").replace("/", "")
            filename = f"xui_backup_{panel_name_part}_{timestamp}.db"

            await context.bot.send_document(
                chat_id=CHANNEL_ID,
                document=backup_data,
                filename=filename,
                caption=f"✅ بکاپ موفق از پنل:\n{panel_url}\nزمان: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            logger.info(f"Backup sent to channel for panel ID {panel_id}")

            with self.db_conn:
                self.db_conn.execute(
                    "UPDATE panels SET last_backup = strftime('%Y-%m-%d %H:%M:%S', 'now', 'localtime') WHERE id = ?",
                    (panel_id,),
                )
            logger.info(f"Updated last_backup time for panel ID {panel_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup panel ID {panel_id} ({panel_url}): {str(e)}", exc_info=True)
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_CHAT_ID,
                    text=f"⚠️ خطا در بکاپ‌گیری پنل ID {panel_id}:\n{panel_url}\n\nخطا:\n`{str(e)}`",
                    parse_mode='Markdown'
                )
            except Exception as send_err:
                 logger.error(f"Failed to send error message to admin: {send_err}")
            return False

    async def auto_backup(self, context: ContextTypes.DEFAULT_TYPE, panel_id=None):
        if not self.backup_lock.acquire(blocking=False):
            return

        try:
            # گرفتن لیست پنل‌های فعال
            cursor = self.db_conn.cursor()
            if panel_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM panels WHERE id = ? AND is_active = 1",
                    (panel_id,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM panels WHERE is_active = 1")
            active_panels_count = cursor.fetchone()[0]

            # ارسال پیام شروع به کانال
            start_message = (
                f"🔄 شروع بکاپ‌گیری خودکار\n"
                f"📊 تعداد پنل‌های فعال: {active_panels_count}\n"
                f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await context.bot.send_message(chat_id=CHANNEL_ID, text=start_message)

            # گرفتن لیست همه ادمین‌ها
            cursor.execute("SELECT user_id FROM admins")
            admin_ids = [row[0] for row in cursor.fetchall()]

            if panel_id:
                for admin_id in admin_ids:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🔄 شروع بکاپ‌گیری از پنل {panel_id}..."
                    )
            else:
                for admin_id in admin_ids:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=f"🔄 شروع بکاپ‌گیری از {active_panels_count} پنل فعال..."
                    )

            # انجام عملیات بکاپ
            success_count = 0
            fail_count = 0
            
            with self.db_conn:
                cursor = self.db_conn.cursor()
                if panel_id:
                    cursor.execute(
                        "SELECT id, panel_url, username, password FROM panels WHERE id = ? AND is_active = 1",
                        (panel_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT id, panel_url, username, password FROM panels WHERE is_active = 1"
                    )
                panels_to_backup = cursor.fetchall()

            if not panels_to_backup:
                message = "⚠️ هیچ پنل فعالی برای بکاپ‌گیری یافت نشد."
                await context.bot.send_message(chat_id=CHANNEL_ID, text=message)
                for admin_id in admin_ids:
                    await context.bot.send_message(chat_id=admin_id, text=message)
                return

            for panel in panels_to_backup:
                if await self.backup_single_panel(*panel, context):
                    success_count += 1
                else:
                    fail_count += 1

            # ارسال نتیجه به کانال
            end_message = (
                f"✅ پایان بکاپ‌گیری خودکار\n"
                f"📊 تعداد کل پنل‌ها: {active_panels_count}\n"
                f"✅ موفق: {success_count}\n"
                f"❌ ناموفق: {fail_count}\n"
                f"⏰ زمان: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            await context.bot.send_message(chat_id=CHANNEL_ID, text=end_message)
            
            # ارسال نتیجه به همه ادمین‌ها
            result_message = (
                f"✅ عملیات بکاپ‌گیری به پایان رسید:\n"
                f"• موفق: {success_count}\n"
                f"• ناموفق: {fail_count}"
            )
            
            for admin_id in admin_ids:
                await context.bot.send_message(
                    chat_id=admin_id,
                    text=result_message
                )

        except Exception as e:
            error_message = f"❌ خطا در فرآیند بکاپ‌گیری:\n{str(e)}"
            logger.error(error_message, exc_info=True)
            # ارسال خطا به کانال و همه ادمین‌ها
            await context.bot.send_message(chat_id=CHANNEL_ID, text=error_message)
            for admin_id in admin_ids:
                try:
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text=error_message
                    )
                except Exception as send_err:
                    logger.error(f"Failed to send error message to admin {admin_id}: {send_err}")

        finally:
            if self.backup_lock.locked():
                self.backup_lock.release()

    async def send_main_menu(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id_to_edit: int | None = None):
        keyboard = [
            [KeyboardButton("➕ افزودن پنل جدید"), KeyboardButton("📋 لیست پنل‌ها")],
            [KeyboardButton("🔄 بکاپ فوری همه"), KeyboardButton("⚙️ تنظیمات")]
        ]
        
        # اضافه کردن دکمه مدیریت ادمین‌ها فقط برای صاحب ربات
        if chat_id == self.OWNER_ID:
            keyboard.append([KeyboardButton("👥 مدیریت ادمین‌ها")])
        
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        text = "🔹 به ربات مدیریت بکاپ X-UI خوش آمدید!\n\nلطفاً یکی از گزینه‌های زیر را انتخاب کنید:"

        try:
            if message_id_to_edit:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id_to_edit,
                    text=text,
                    reply_markup=reply_markup
                )
                logger.debug(f"Edited message {message_id_to_edit} in chat {chat_id} with main menu.")
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup
                )
                logger.debug(f"Sent main menu to chat {chat_id}.")
        except Exception as e:
             logger.error(f"Error in send_main_menu (chat_id: {chat_id}, target_message_id: {message_id_to_edit}): {str(e)}", exc_info=True)
             try:
                 await context.bot.send_message(
                     chat_id,
                     "⚠️ خطا در نمایش منو. لطفا /start را مجددا وارد کنید.",
                 )
             except Exception as send_err:
                 logger.error(f"Failed to send error message about menu display: {send_err}")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id
        
        # بررسی ادمین بودن کاربر
        if not await self.is_admin(user_id):
            await update.message.reply_text(
                "⛔️ شما به این ربات دسترسی ندارید!\n\n"
                "💬 جهت درخواست دسترسی با پشتیبانی در ارتباط باشید:\n"
                "👤 @ali_sudo_of"
            )
            return ConversationHandler.END
        
        # اگر کاربر ادمین باشد، ادامه روند عادی
        context.user_data.clear()
        await self.send_main_menu(context, chat_id=update.message.chat_id)
        return ConversationHandler.END

    # --- تابع button_handler با اصلاحات ---
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """هندلر برای تمام CallbackQuery ها."""
        query = update.callback_query
        if not query:
            logger.warning("button_handler received an update without a callback_query")
            return ConversationHandler.END

        try:
            await query.answer()
        except Exception as e:
            logger.warning(f"Failed to answer callback query: {e}")

        query_data = query.data
        chat_id = query.message.chat.id
        message_id = query.message.message_id

        logger.debug(f"Button pressed: {query_data} in chat {chat_id}")

        # --- خارج از conversation handlers ---
        if query_data == "back_to_menu":
            context.user_data.clear()
            await self.send_main_menu(context, chat_id=chat_id, message_id_to_edit=message_id)
            return ConversationHandler.END

        elif query_data == "list_panels":
            await self.show_panels_list(query, context)
            return ConversationHandler.END

        elif query_data == "force_backup_all":
            try:
                await query.edit_message_text("🔄 در حال انجام بکاپ فوری برای تمام پنل‌های فعال...")
                await self.auto_backup(context)
                await query.edit_message_text("✅ درخواست بکاپ فوری ارسال شد. نتایج به کانال ارسال می‌شود.")
            except Exception as e:
                logger.error(f"Force backup failed: {str(e)}", exc_info=True)
                await query.edit_message_text(f"❌ خطا در شروع بکاپ فوری:\n{str(e)}")
            return ConversationHandler.END

        elif query_data == "settings":
            await self.show_settings(query, context)
            return ConversationHandler.END

        elif query_data.startswith("toggle_"):
            panel_id = int(query_data.split("_")[1])
            await self.toggle_panel_status(query, panel_id, context)
            return ConversationHandler.END

        # --- سایر دکمه‌ها با هماهنگی با conversation handlers ---
        elif query_data == "add_panel":
            await query.edit_message_text(
                "لطفاً آدرس کامل پنل را ارسال کنید (مثال: http://example.com:8080):"
            )
            return PANEL_URL

        elif query_data == "set_backup_interval":
            current_interval = self._get_setting("backup_interval", DEFAULT_BACKUP_INTERVAL)
            await query.edit_message_text(
                f"فاصله بکاپ‌گیری فعلی: {current_interval} دقیقه.\n"
                f"لطفاً فاصله جدید را به دقیقه وارد کنید (حداقل {MIN_BACKUP_INTERVAL} دقیقه):"
            )
            return SET_BACKUP_INTERVAL

        elif query_data.startswith("delete_"):
            panel_id_to_delete = int(query_data.split("_")[1])
            context.user_data['panel_to_delete'] = panel_id_to_delete
            keyboard = [
                [
                    KeyboardButton(f"✅ بله، حذف کن (پنل {panel_id_to_delete})"),
                    KeyboardButton("❌ خیر، انصراف")
                ]
            ]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            await query.edit_message_text(
                f"⚠️ آیا از حذف پنل {panel_id_to_delete} مطمئن هستید؟ این عمل قابل بازگشت نیست.",
                reply_markup=reply_markup
            )
            return CONFIRM_DELETE

        elif query_data == "confirm_add":
            await self.confirm_panel_addition(query, context)
            return ConversationHandler.END

        elif query_data == "cancel_add":
            await query.edit_message_text("❌ عملیات افزودن پنل لغو شد.")
            context.user_data.clear()
            await self.send_main_menu(context, chat_id=chat_id, message_id_to_edit=message_id)
            return ConversationHandler.END

        else:
            logger.warning(f"Unhandled button data in button_handler: {query_data}")
            await self.send_main_menu(context, chat_id=chat_id, message_id_to_edit=message_id)
            return ConversationHandler.END

    # --- پایان تابع button_handler ---


    async def show_panels_list(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            with self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "SELECT id, panel_url, last_backup, is_active FROM panels ORDER BY created_at DESC"
                )
                panels = cursor.fetchall()

            if isinstance(update, Update):
                # اگر از message آمده
                message_obj = update.message
            else:
                # اگر از callback_query آمده
                message_obj = update.message

            if not panels:
                keyboard = [[KeyboardButton("🔙 بازگشت به منو")]]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                await message_obj.reply_text(
                    "⚠️ هیچ پنلی ثبت نشده است!", 
                    reply_markup=reply_markup
                )
                return

            message = "📋 لیست پنل‌های ثبت شده:\n\n"
            keyboard = []

            for panel in panels:
                panel_id, panel_url, last_backup_ts, is_active = panel
                status_icon = "✅" if is_active else "❌"
                status_text = "فعال" if is_active else "غیرفعال"
                
                message += (
                    f"🆔 پنل: {panel_id}\n"
                    f"🔗 آدرس: {panel_url}\n"
                    f"⏱ آخرین بکاپ: {last_backup_ts or 'هنوز انجام نشده'}\n"
                    f"{status_icon} وضعیت: {status_text}\n"
                    f"--------------------\n"
                )
                keyboard.append([
                    KeyboardButton(f"{'🔴' if is_active else '🟢'} تغییر وضعیت پنل {panel_id}"),
                    KeyboardButton(f"🗑 حذف پنل {panel_id}")
                ])

            keyboard.append([KeyboardButton("🔙 بازگشت به منو")])
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

            await message_obj.reply_text(message, reply_markup=reply_markup)

        except Exception as e:
            logger.error(f"Error in show_panels_list: {str(e)}", exc_info=True)
            try:
                if isinstance(update, Update):
                    # اگر از message آمده
                    await update.message.reply_text("⚠️ خطا در نمایش لیست پنل‌ها.")
                else:
                    # اگر از callback_query آمده
                    await update.edit_message_text("⚠️ خطا در نمایش لیست پنل‌ها.")
            except Exception:
                logger.error("Failed to send error message in show_panels_list")

    async def toggle_panel_status(self, update: Update, panel_id: int, context: ContextTypes.DEFAULT_TYPE):
        try:
            with self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute(
                    "SELECT is_active FROM panels WHERE id = ?", (panel_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    if isinstance(update, Update):
                        await update.message.reply_text("⚠️ پنل مورد نظر یافت نشد!")
                    else:
                        await update.edit_message_text("⚠️ پنل مورد نظر یافت نشد!")
                    return

                new_status = not result[0]
                cursor.execute(
                    "UPDATE panels SET is_active = ? WHERE id = ?",
                    (new_status, panel_id)
                )

            status_text = "فعال" if new_status else "غیرفعال"
            
            if isinstance(update, Update):
                # اگر از message آمده
                await update.message.reply_text(
                    f"✅ وضعیت پنل {panel_id} با موفقیت به {status_text} تغییر یافت."
                )
                await self.show_panels_list(update, context)
            else:
                # اگر از callback_query آمده
                await update.edit_message_text(
                    f"✅ وضعیت پنل {panel_id} با موفقیت به {status_text} تغییر یافت."
                )
                # برگشت به منوی اصلی
                await self.send_main_menu(context, chat_id=update.message.chat.id)

        except Exception as e:
            logger.error(f"Error in toggle_panel_status: {str(e)}")
            if isinstance(update, Update):
                await update.message.reply_text("❌ خطا در تغییر وضعیت پنل!")
            else:
                await update.edit_message_text("❌ خطا در تغییر وضعیت پنل!")

    # --- تابع حذف واقعی پنل ---
    async def _perform_delete(self, update: Update, panel_id: int, context: ContextTypes.DEFAULT_TYPE):
        try:
            with self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute("DELETE FROM panels WHERE id = ?", (panel_id,))
                
            if isinstance(update, Update):
                # اگر از message آمده
                await update.message.reply_text(
                    f"✅ پنل {panel_id} با موفقیت حذف شد."
                )
                await self.show_panels_list(update, context)
            else:
                # اگر از callback_query آمده
                await update.edit_message_text(
                    f"✅ پنل {panel_id} با موفقیت حذف شد."
                )
                # برگشت به منوی اصلی
                await self.send_main_menu(context, chat_id=update.message.chat.id)

        except Exception as e:
            logger.error(f"Error in _perform_delete: {str(e)}")
            if isinstance(update, Update):
                await update.message.reply_text("❌ خطا در حذف پنل!")
            else:
                await update.edit_message_text("❌ خطا در حذف پنل!")

    async def show_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            with self.db_conn:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM panels WHERE is_active = 1")
                active_panels = cursor.fetchone()[0]
                backup_interval_minutes = self._get_setting("backup_interval", DEFAULT_BACKUP_INTERVAL)

            keyboard = [
                [KeyboardButton("⏱ تغییر فاصله بکاپ")],  # حذف نمایش فاصله فعلی از دکمه
                [KeyboardButton("🔙 بازگشت به منو")]
            ]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

            await update.message.reply_text(
                f"⚙️ تنظیمات ربات:\n\n"
                f"• فاصله بکاپ‌گیری خودکار: هر {backup_interval_minutes} دقیقه\n"
                f"• تعداد پنل‌های فعال: {active_panels}",
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Error in show_settings: {str(e)}", exc_info=True)
            await update.message.reply_text("⚠️ خطا در نمایش تنظیمات.")


    # --- Handlers for Conversation ---

    async def handle_panel_url(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        # ... (بدون تغییر) ...
        panel_url = update.message.text.strip()
        if not panel_url:
             await update.message.reply_text("⚠️ آدرس پنل نمی‌تواند خالی باشد. لطفاً دوباره وارد کنید:")
             return PANEL_URL

        if not (panel_url.startswith("http://") or panel_url.startswith("https://")):
             await update.message.reply_text(
                 "⚠️ آدرس پنل باید با `http://` یا `https://` شروع شود. لطفاً آدرس کامل را وارد کنید:" , parse_mode='Markdown'
             )
             return PANEL_URL

        context.user_data["panel_url"] = panel_url
        await update.message.reply_text("🔸 نام کاربری ادمین پنل را وارد کنید:")
        return USERNAME

    async def handle_username(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        # ... (بدون تغییر) ...
        username = update.message.text.strip()
        if not username:
             await update.message.reply_text("⚠️ نام کاربری نمی‌تواند خالی باشد. لطفاً دوباره وارد کنید:")
             return USERNAME

        context.user_data["username"] = username
        await update.message.reply_text("🔒 رمز عبور ادمین پنل را وارد کنید:")
        return PASSWORD

    async def handle_password(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> int:
        password = update.message.text
        if not password:
            await update.message.reply_text("⚠️ رمز عبور نمی‌تواند خالی باشد. لطفاً دوباره وارد کنید:")
            return PASSWORD

        context.user_data["password"] = password
        hidden_password = "•" * len(password)

        # استفاده از دکمه‌های معمولی به جای شیشه‌ای
        keyboard = [
            [KeyboardButton("✅ تأیید و ثبت"), KeyboardButton("❌ انصراف")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

        await update.message.reply_text(
            f"👇 اطلاعات زیر ثبت خواهد شد:\n\n"
            f"🔗 آدرس: {context.user_data['panel_url']}\n"
            f"👤 یوزرنیم: {context.user_data['username']}\n"
            f"🔑 پسورد: {hidden_password}\n\n"
            f"❓ آیا اطلاعات صحیح است و می‌خواهید این پنل را اضافه کنید؟",
            reply_markup=reply_markup
        )
        return CONFIRMATION

    async def handle_confirmation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        text = update.message.text

        if text == "✅ تأیید و ثبت":
            try:
                panel_data = context.user_data
                if not all(k in panel_data for k in ("panel_url", "username", "password")):
                    await update.message.reply_text("⚠️ اطلاعات پنل ناقص است. لطفاً از ابتدا شروع کنید.")
                    await self.send_main_menu(context, chat_id=update.message.chat_id)
                    return ConversationHandler.END

                with self.db_conn:
                    cursor = self.db_conn.cursor()
                    cursor.execute(
                        "INSERT INTO panels (panel_url, username, password) VALUES (?, ?, ?)",
                        (
                            panel_data["panel_url"],
                            panel_data["username"],
                            panel_data["password"],
                        ),
                    )
                    panel_id = cursor.lastrowid

                await update.message.reply_text(f"✅ پنل با ID {panel_id} با موفقیت ثبت شد!\nدر حال تلاش برای گرفتن اولین بکاپ...")
                await self.auto_backup(context, panel_id=panel_id)

            except sqlite3.IntegrityError:
                await update.message.reply_text("⚠️ این آدرس پنل قبلاً ثبت شده است!")
            except Exception as e:
                await update.message.reply_text(f"❌ خطا در ثبت پنل: {str(e)}")
            finally:
                context.user_data.clear()
                await self.send_main_menu(context, chat_id=update.message.chat_id)
                return ConversationHandler.END

        elif text == "❌ انصراف":
            context.user_data.clear()
            await update.message.reply_text("❌ عملیات افزودن پنل لغو شد.")
            await self.send_main_menu(context, chat_id=update.message.chat_id)
            return ConversationHandler.END

        return CONFIRMATION

    async def handle_backup_interval(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            text = update.message.text.strip()
            
            if text == "🔙 بازگشت به منو":
                await self.send_main_menu(context, chat_id=update.message.chat_id)
                return ConversationHandler.END
            
            try:
                interval_minutes = int(text)
            except ValueError:
                await update.message.reply_text(
                    "⚠️ لطفاً فقط عدد وارد کنید.\n"
                    "مثال: 15"
                )
                return SET_BACKUP_INTERVAL
            
            if interval_minutes < MIN_BACKUP_INTERVAL:
                await update.message.reply_text(
                    f"⚠️ فاصله بکاپ‌گیری نباید کمتر از {MIN_BACKUP_INTERVAL} دقیقه باشد.\n"
                    "لطفاً عدد بزرگتری وارد کنید:"
                )
                return SET_BACKUP_INTERVAL

            # ذخیره تنظیمات جدید
            self._set_setting("backup_interval", interval_minutes)
            self.backup_interval_seconds = interval_minutes * 60

            # بروزرسانی job queue
            if hasattr(context, 'job_queue'):
                current_jobs = context.job_queue.get_jobs_by_name("auto_backup_job")
                for job in current_jobs:
                    job.schedule_removal()

                context.job_queue.run_repeating(
                    self.auto_backup,
                    interval=self.backup_interval_seconds,
                    first=10,
                    name="auto_backup_job"
                )

            # ارسال پیام موفقیت و بازگشت به منوی تنظیمات
            keyboard = [
                [KeyboardButton("⏱ تغییر فاصله بکاپ")],
                [KeyboardButton("🔙 بازگشت به منو")]
            ]
            reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
            
            success_message = (
                f"✅ تنظیمات جدید با موفقیت ذخیره شد:\n\n"
                f"• فاصله بکاپ‌گیری: هر {interval_minutes} دقیقه\n"
                f"• وضعیت: فعال"
            )
            
            await update.message.reply_text(success_message, reply_markup=reply_markup)
            await self.show_settings(update, context)
            return ConversationHandler.END

        except Exception as e:
            logger.error(f"Error in handle_backup_interval: {str(e)}")
            await update.message.reply_text(
                "❌ خطا در تنظیم فاصله بکاپ‌گیری!\n"
                "لطفاً دوباره تلاش کنید."
            )
            await self.send_main_menu(context, chat_id=update.message.chat_id)
            return ConversationHandler.END

    async def cancel_conversation(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        # ... (بدون تغییر) ...
        user = update.effective_user
        logger.info(f"User {user.id} canceled the conversation.")
        context.user_data.clear()
        await update.message.reply_text('عملیات لغو شد.')
        await self.send_main_menu(context, chat_id=update.message.chat_id)
        return ConversationHandler.END

    # --- Error Handler ---
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        # ... (بدون تغییر) ...
        logger.error(f"Exception while handling an update: {context.error}", exc_info=context.error)
        if isinstance(update, Update) and update.effective_chat:
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="⚠️ متاسفانه خطایی در پردازش درخواست شما رخ داد. لطفاً دوباره تلاش کنید یا با ادمین تماس بگیرید.",
                )
            except Exception as e:
                logger.error(f"Failed to send error message to user: {e}")

    async def handle_button_press(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = update.message.text
        user_id = update.effective_user.id
        
        # بررسی ادمین بودن کاربر
        if not await self.is_admin(user_id):
            await update.message.reply_text(
                "⛔️ شما به این ربات دسترسی ندارید!\n\n"
                "💬 جهت درخواست دسترسی با پشتیبانی در ارتباط باشید:\n"
                "👤 @ali_sudo_of"
            )
            return ConversationHandler.END

        # --- پردازش دکمه‌های تایید و انصراف حذف ---
        if text.startswith("✅ بله، حذف کن"):
            panel_id = context.user_data.get('panel_to_delete')
            if panel_id:
                await self._perform_delete(update, panel_id, context)
                context.user_data.pop('panel_to_delete', None)
            else:
                await update.message.reply_text("⚠️ خطا: پنل مورد نظر برای حذف مشخص نیست.")
            
            await self.send_main_menu(context, chat_id=update.message.chat_id)
            return ConversationHandler.END
            
        elif text == "❌ خیر، انصراف":
            context.user_data.pop('panel_to_delete', None)
            await update.message.reply_text("عملیات حذف لغو شد.")
            await self.send_main_menu(context, chat_id=update.message.chat_id)
            return ConversationHandler.END

        # --- مدیریت ادمین‌ها (مخصوص صاحب ربات) ---
        if text in ["👥 مدیریت ادمین‌ها", "➕ افزودن ادمین جدید", "❌ حذف ادمین"]:
            if not await self.is_owner(user_id):
                await update.message.reply_text("⚠️ این بخش فقط برای صاحب ربات در دسترس است!")
                await self.send_main_menu(context, chat_id=update.message.chat_id)
                return ConversationHandler.END
            
            if text == "👥 مدیریت ادمین‌ها":
                await self.list_admins(update, context)
                return ConversationHandler.END
            elif text == "➕ افزودن ادمین جدید":
                keyboard = [[KeyboardButton("🔙 بازگشت به منو")]]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                await update.message.reply_text(
                    "لطفاً آیدی عددی کاربر مورد نظر را وارد کنید:",
                    reply_markup=reply_markup
                )
                return ADD_ADMIN
            elif text == "❌ حذف ادمین":
                keyboard = [[KeyboardButton("🔙 بازگشت به منو")]]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                await update.message.reply_text(
                    "لطفاً آیدی عددی ادمین مورد نظر را برای حذف وارد کنید:",
                    reply_markup=reply_markup
                )
                return REMOVE_ADMIN

        # --- دکمه‌های عمومی برای همه ادمین‌ها ---
        if text == "📋 لیست پنل‌ها":
            await self.show_panels_list(update, context)
            return

        elif text == "🔄 بکاپ فوری همه":
            await update.message.reply_text("🔄 در حال انجام بکاپ فوری از تمام پنل‌های فعال...")
            await self.auto_backup(context)
            return

        elif text == "⚙️ تنظیمات":
            await self.show_settings(update, context)
            return

        elif text == "➕ افزودن پنل جدید":
            await update.message.reply_text("لطفاً آدرس کامل پنل را ارسال کنید (مثال: http://example.com:8080):")
            return PANEL_URL

        elif text == "⏱ تغییر فاصله بکاپ":
            current_interval = self._get_setting("backup_interval", DEFAULT_BACKUP_INTERVAL)
            await update.message.reply_text(
                f"فاصله بکاپ‌گیری فعلی: {current_interval} دقیقه.\n"
                f"لطفاً فاصله جدید را به دقیقه وارد کنید (حداقل {MIN_BACKUP_INTERVAL} دقیقه):"
            )
            return SET_BACKUP_INTERVAL

        elif text == "🔙 بازگشت به منو":
            await self.send_main_menu(context, chat_id=update.message.chat_id)
            return ConversationHandler.END

        # برای دکمه‌های تغییر وضعیت و حذف پنل
        elif text.startswith("🔴") or text.startswith("🟢"):  # تغییر وضعیت پنل
            try:
                panel_id = int(text.split("پنل")[1].strip())
                await self.toggle_panel_status(update, panel_id, context)
                return ConversationHandler.END
            except (ValueError, IndexError):
                await update.message.reply_text("❌ خطا در پردازش درخواست!")
                return ConversationHandler.END

        elif text.startswith("🗑 حذف پنل"):
            try:
                panel_id = int(text.split("پنل")[1].strip())
                context.user_data['panel_to_delete'] = panel_id
                keyboard = [
                    [KeyboardButton(f"✅ بله، حذف کن (پنل {panel_id})"), KeyboardButton("❌ خیر، انصراف")]
                ]
                reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
                await update.message.reply_text(
                    f"⚠️ آیا از حذف پنل {panel_id} مطمئن هستید؟ این عمل قابل بازگشت نیست.",
                    reply_markup=reply_markup
                )
                return CONFIRM_DELETE
            except (ValueError, IndexError):
                await update.message.reply_text("❌ خطا در پردازش درخواست!")
            return

        # اگر به هیچ کدام از شرایط بالا نرسید
        logger.warning(f"Unhandled button text: {text}")
        await self.send_main_menu(context, chat_id=update.message.chat_id)
        return ConversationHandler.END

    async def is_admin(self, user_id: int) -> bool:
        """بررسی ادمین بودن کاربر"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT 1 FROM admins WHERE user_id = ?", (user_id,))
        return cursor.fetchone() is not None

    async def is_owner(self, user_id: int) -> bool:
        """بررسی صاحب ربات بودن کاربر"""
        return user_id == self.OWNER_ID

    async def add_admin(self, admin_id: int, added_by: int) -> bool:
        """اضافه کردن ادمین جدید"""
        try:
            with self.db_conn:
                self.db_conn.execute(
                    "INSERT INTO admins (user_id, added_by) VALUES (?, ?)",
                    (admin_id, added_by)
                )
            return True
        except sqlite3.IntegrityError:
            return False

    async def remove_admin(self, admin_id: int) -> bool:
        """حذف ادمین"""
        if admin_id == self.OWNER_ID:
            return False
        try:
            with self.db_conn:
                self.db_conn.execute("DELETE FROM admins WHERE user_id = ?", (admin_id,))
            return True
        except Exception:
            return False

    async def list_admins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """نمایش لیست ادمین‌ها"""
        if not await self.is_owner(update.effective_user.id):
            return
        
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT a.user_id, a.added_at, b.user_id as added_by
            FROM admins a
            LEFT JOIN admins b ON a.added_by = b.user_id
            ORDER BY a.added_at DESC
        """)
        admins = cursor.fetchall()
        
        message = "👥 لیست ادمین‌های ربات:\n\n"
        for admin in admins:
            user_id, added_at, added_by = admin
            is_owner = "👑" if user_id == self.OWNER_ID else "👤"
            message += f"{is_owner} آیدی: {user_id}\n"
            if user_id != self.OWNER_ID:
                message += f"📅 تاریخ افزودن: {added_at}\n"
                message += f"➕ اضافه شده توسط: {added_by}\n"
            message += "──────────────\n"
        
        keyboard = [
            [KeyboardButton("➕ افزودن ادمین جدید")],
            [KeyboardButton("❌ حذف ادمین")],
            [KeyboardButton("🔙 بازگشت به منو")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(message, reply_markup=reply_markup)

    async def handle_add_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """پردازش درخواست اضافه کردن ادمین جدید"""
        if not await self.is_owner(update.effective_user.id):
            await update.message.reply_text("⚠️ شما دسترسی به این بخش را ندارید!")
            return ConversationHandler.END

        try:
            new_admin_id = int(update.message.text.strip())
            
            # بررسی اینکه آیا قبلاً ادمین است
            if await self.is_admin(new_admin_id):
                await update.message.reply_text("⚠️ این کاربر در حال حاضر ادمین است!")
                await self.list_admins(update, context)
                return ConversationHandler.END

            # اضافه کردن ادمین جدید
            if await self.add_admin(new_admin_id, update.effective_user.id):
                try:
                    # ارسال پیام به ادمین جدید
                    await context.bot.send_message(
                        chat_id=new_admin_id,
                        text="✅ شما به عنوان ادمین ربات بکاپ X-UI منصوب شدید."
                    )
                except Exception as e:
                    logger.warning(f"Could not send message to new admin: {e}")

                await update.message.reply_text(f"✅ کاربر {new_admin_id} با موفقیت به لیست ادمین‌ها اضافه شد.")
            else:
                await update.message.reply_text("❌ خطا در اضافه کردن ادمین جدید!")

        except ValueError:
            await update.message.reply_text("⚠️ لطفاً یک آیدی عددی معتبر وارد کنید!")
            return ADD_ADMIN

        await self.list_admins(update, context)
        return ConversationHandler.END

    async def handle_remove_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """پردازش درخواست حذف ادمین"""
        if not await self.is_owner(update.effective_user.id):
            await update.message.reply_text("⚠️ شما دسترسی به این بخش را ندارید!")
            return ConversationHandler.END

        try:
            admin_id = int(update.message.text.strip())
            
            # نمی‌توان صاحب ربات را حذف کرد
            if admin_id == self.OWNER_ID:
                await update.message.reply_text("⚠️ نمی‌توانید صاحب ربات را حذف کنید!")
                await self.list_admins(update, context)
                return ConversationHandler.END

            # بررسی اینکه آیا کاربر ادمین است
            if not await self.is_admin(admin_id):
                await update.message.reply_text("⚠️ این کاربر ادمین نیست!")
                await self.list_admins(update, context)
                return ConversationHandler.END

            # حذف ادمین
            if await self.remove_admin(admin_id):
                try:
                    # ارسال پیام به ادمین حذف شده
                    await context.bot.send_message(
                        chat_id=admin_id,
                        text="⚠️ دسترسی ادمین شما در ربات بکاپ X-UI لغو شد."
                    )
                except Exception as e:
                    logger.warning(f"Could not send message to removed admin: {e}")

                await update.message.reply_text(f"✅ کاربر {admin_id} با موفقیت از لیست ادمین‌ها حذف شد.")
            else:
                await update.message.reply_text("❌ خطا در حذف ادمین!")

        except ValueError:
            await update.message.reply_text("⚠️ لطفاً یک آیدی عددی معتبر وارد کنید!")
            return REMOVE_ADMIN

        await self.list_admins(update, context)
        return ConversationHandler.END

    def run(self):
        """شروع به کار ربات."""
        application = Application.builder().token(BOT_TOKEN).build()
        application.add_error_handler(self.error_handler)

        # --- Conversation Handler ---
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("start", self.start),
                MessageHandler(filters.Regex("^➕ افزودن پنل جدید$"), self.handle_panel_url),
                MessageHandler(filters.Regex("^⏱ تغییر فاصله بکاپ$"), 
                    lambda update, context: self.handle_backup_interval(update, context)),
                MessageHandler(filters.Regex("^🗑 حذف پنل"), self.handle_button_press),
                MessageHandler(filters.Regex("^👥 مدیریت ادمین‌ها$"), self.handle_button_press),
                MessageHandler(filters.Regex("^➕ افزودن ادمین جدید$"), self.handle_button_press),
                MessageHandler(filters.Regex("^❌ حذف ادمین$"), self.handle_button_press),
                CallbackQueryHandler(self.button_handler)
            ],
            states={
                PANEL_URL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_panel_url)],
                USERNAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_username)],
                PASSWORD: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_password)],
                CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_confirmation)],
                SET_BACKUP_INTERVAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_backup_interval)],
                CONFIRM_DELETE: [
                    MessageHandler(filters.Regex("^✅ بله، حذف کن"), self.handle_button_press),
                    MessageHandler(filters.Regex("^❌ خیر، انصراف$"), self.handle_button_press)
                ],
                ADD_ADMIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_add_admin)],
                REMOVE_ADMIN: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_remove_admin)]
            },
            fallbacks=[
                CommandHandler("cancel", self.cancel_conversation),
                CommandHandler("start", self.start),
                MessageHandler(filters.Regex("^🔙 بازگشت به منو$"), 
                    lambda update, context: self.send_main_menu(context, update.message.chat_id)),
                CallbackQueryHandler(self.button_handler)
            ],
            allow_reentry=True,
            name="main_conversation"
        )

        # اضافه کردن handlers
        application.add_handler(conv_handler)
        
        # هندلر برای دکمه‌های عمومی (که در conversation handler پوشش داده نشده‌اند)
        application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self.handle_button_press
        ))

        # --- Job Queue ---
        application.job_queue.run_repeating(
            self.auto_backup,
            interval=self.backup_interval_seconds,
            first=10,
            name="auto_backup_job"
        )
        logger.info(f"Initial auto-backup job scheduled with interval {self.backup_interval_seconds} seconds.")

        # --- Start Polling ---
        logger.info("Starting bot polling...")
        application.run_polling()
        logger.info("Bot polling stopped.")


# --- اجرای ربات ---
if __name__ == "__main__":
    if BOT_TOKEN == "YOUR_BOT_TOKEN" or ADMIN_CHAT_ID == 0 or CHANNEL_ID == "@YOUR_CHANNEL_ID":
        logger.critical("FATAL: BOT_TOKEN, ADMIN_CHAT_ID, or CHANNEL_ID is not set in the script!")
        exit(1)

    bot = XUIBackupBot()
    try:
        bot.run()
    except Exception as e:
        logger.critical(f"Fatal error running the bot: {str(e)}", exc_info=True)
    finally:
        if bot.db_conn:
            bot.db_conn.close()
            logger.info("Database connection closed.")