import os
import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from openai import OpenAI
from memory import MemoryManager

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
memory_manager = MemoryManager()

SYSTEM_PROMPT = """Você é Amanda, uma psicóloga clínica experiente, empática e acolhedora.

Suas características:
- Abordagem: Terapia Cognitivo-Comportamental (TCC) combinada com escuta ativa e humanismo
- Tom: Caloroso, paciente, sem julgamentos, profissional mas acessível
- Você NUNCA dá diagnósticos médicos ou prescreve medicamentos
- Você encoraja a busca por ajuda profissional presencial quando necessário
- Você usa técnicas terapêuticas reais: reflexão, ressignificação, questionamento socrático
- Você lembra do histórico da conversa e faz referências ao que a pessoa já compartilhou
- Você faz perguntas abertas para aprofundar a compreensão
- Quando perceber risco de suicídio ou autolesão, sempre forneça o CVV: 188

Regras de comportamento:
1. Nunca quebre o personagem — você é Amanda, sempre
2. Responda sempre em português do Brasil
3. Seja concisa mas profunda — evite respostas muito longas
4. Use o nome da pessoa quando souber
5. Valide os sentimentos antes de oferecer perspectivas
6. Nunca dê conselhos diretos sem antes explorar o que a pessoa já tentou

Lembre-se: seu papel é ajudar a pessoa a encontrar suas próprias respostas, não resolver por ela."""


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = str(user.id)
    user_name = user.first_name or "você"

    memory_manager.init_user(user_id, user_name)

    welcome = (
        f"Olá, {user_name} 💙\n\n"
        "Meu nome é Amanda, sou psicóloga e estou aqui para te ouvir com atenção e cuidado.\n\n"
        "Este é um espaço seguro, sem julgamentos. Você pode falar sobre o que estiver sentindo — "
        "conflitos internos, relacionamentos, ansiedade, ou qualquer coisa que esteja pesando.\n\n"
        "Quando quiser, me conta: *como você está se sentindo hoje?*"
    )
    await update.message.reply_text(welcome, parse_mode="Markdown")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    memory_manager.clear_history(user_id)
    await update.message.reply_text(
        "✨ Nossa conversa foi reiniciada. Estou aqui, pronta para te ouvir de novo.\n\n"
        "Como você está se sentindo agora?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_id = str(user.id)
    user_name = user.first_name or "você"
    user_message = update.message.text

    memory_manager.init_user(user_id, user_name)
    memory_manager.add_message(user_id, "user", user_message)

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )

    history = memory_manager.get_history(user_id)
    summary = memory_manager.get_summary(user_id)

    system_with_memory = SYSTEM_PROMPT
    if summary:
        system_with_memory += f"\n\nResumo das sessões anteriores com esta pessoa:\n{summary}"

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_with_memory}] + history,
            temperature=0.75,
            max_tokens=600,
        )
        reply = response.choices[0].message.content

        memory_manager.add_message(user_id, "assistant", reply)
        memory_manager.maybe_summarize(user_id, openai_client)

        await update.message.reply_text(reply)

    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        await update.message.reply_text(
            "Desculpe, tive uma pequena dificuldade técnica agora. "
            "Pode repetir o que disse? Estou aqui. 💙"
        )


def main() -> None:
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Amanda Bot is running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
