import os
import base64
import pathlib
from io import BytesIO
from PIL import Image
import requests
from dotenv import load_dotenv
from openai import OpenAI
from openai import BadRequestError, NotFoundError

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def save_png_from_b64(b64_str: str, path: pathlib.Path):
    img_bytes = base64.b64decode(b64_str)
    Image.open(BytesIO(img_bytes)).save(path, format="PNG")


def get_user_prompt() -> str:
    user_prompt = input(
        "Какое изображение сгенерировать? (или нажмите Enter для изображения по умолчанию): ").strip()

    if not user_prompt:
        default_prompt = "a beautiful landscape with mountains and lake during sunset"
        print(f"Используется изображение по умолчанию: '{default_prompt}'")
        return default_prompt

    return user_prompt


def generate_and_save_image(prompt: str) -> str:
    try:
        resp = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        output_dir = pathlib.Path("output")
        output_dir.mkdir(exist_ok=True)

        prompt_hash = hash(prompt) % 1000000
        filename = f"generated_image_{prompt_hash}.png"
        out_path = output_dir / filename

        if hasattr(resp.data[0], "b64_json") and resp.data[0].b64_json:
            save_png_from_b64(resp.data[0].b64_json, out_path)
            print(f"✓ Изображение сохранено из base64 данных")

        elif hasattr(resp.data[0], "url") and resp.data[0].url:
            url = resp.data[0].url
            png = requests.get(url, timeout=60).content
            out_path.write_bytes(png)
            print(f"✓ Изображение скачано и сохранено по URL")

        else:
            raise ValueError("Неожиданный формат ответа от API")

        return str(out_path.resolve())

    except BadRequestError as e:
        print(f"✗ Ошибка запроса: {e}")
        print("Проверьте промпт - возможно, он содержит запрещенный контент.")
        return None
    except NotFoundError as e:
        print(f"✗ Ошибка: {e}")
        print("Проверьте доступность модели DALL-E 3 в вашем аккаунте.")
        return None
    except Exception as e:
        print(f"✗ Неожиданная ошибка: {repr(e)}")
        return None


def main():
    print("🎨 ГЕНЕРАТОР ИЗОБРАЖЕНИЙ DALL-E 3")
    print("=" * 50)

    prompt = get_user_prompt()

    print("\n🔄 Генерируем изображение...")

    file_path = generate_and_save_image(prompt)

    if file_path:
        print("\n" + "=" * 50)
        print("✅ ИЗОБРАЖЕНИЕ УСПЕШНО СОЗДАНО!")
        print("=" * 50)
        print(f"Использованный промпт: '{prompt}'")
        print(f"Путь к файлу: {file_path}")

        try:
            Image.open(file_path).show()
            print("✓ Изображение открыто для просмотра")
        except Exception as e:
            print(f"⚠ Не удалось открыть изображение для просмотра: {e}")
    else:
        print("\n❌ Не удалось создать изображение")


if __name__ == "__main__":
    main()