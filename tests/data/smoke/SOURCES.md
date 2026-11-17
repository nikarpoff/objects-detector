# Источники снимков для test_model_smoke

Снимки взяты с Wikimedia Commons (проверено 2026-11-17), все под CC0 1.0. Проверка AC-54: модель класса nano находит человека на `person_path.jpg` и стул или стол на `room_chairs.jpg`, людей на нём нет.

| Файл | Что на снимке | Источник | Автор | Лицензия | Изменения |
| --- | --- | --- | --- | --- | --- |
| `person_path.jpg` | женщина идёт по дорожке, впереди и близко | [File:Woman walking down path (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Woman_walking_down_path_(Unsplash).jpg) | Biel Morro (Unsplash, bielmb) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | миниатюра Commons 960 пикселей уменьшена до 640x425 и пережата в JPEG (Pillow, качество 85, без EXIF) |
| `room_chairs.jpg` | переговорная: стол и стулья, людей нет | [File:Chairs in a meeting room (Unsplash).jpg](https://commons.wikimedia.org/wiki/File:Chairs_in_a_meeting_room_(Unsplash).jpg) | Breather (Unsplash) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | побайтная копия `testdata/minidataset/images/img002.jpg` из `assistant-low-vision-platform` (640x427) |
