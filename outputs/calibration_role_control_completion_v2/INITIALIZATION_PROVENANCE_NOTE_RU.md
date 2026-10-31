# Ограничение initialization seed и доступная provenance

Независимая проверка исходного frozen-кода (`SEED_ORDER_REVIEW_20261020.json`) показала: `ECGResNetSmall()` конструировалась в `control.py` на строке 109, а вызовы `random.seed`, `numpy.random.seed`, `torch.manual_seed` и `torch.cuda.manual_seed_all` выполнялись на строке 111. То есть declared seed задавал последующую случайность обучения и augmentations, но не initial parameters.

Контрпример уже сохранён review: при одном fit seed разные ambient Torch RNG дали разные хеши начальных параметров; установка seed до constructor дала одинаковые веса. Это ограничение воспроизводимости начальной точки, а не признак ошибки A/B split и не доказательство неверных saved predictions.

Доступная provenance включает fit config/history, hashes checkpoint и replay лучших checkpoint, плюс resume-state, который сохраняет RNG во время обучения. В checkpoint `state_dict` — обученные веса. Отдельного initial RNG snapshot или initial parameter hash до первого optimizer step в frozen fit provenance не найдено; поэтому исходные веса восстановить по declared seed нельзя. Никакой подбор RNG, перебор или повторное обучение не проводились.

Все девять сохранённых fits остаются валидными как фиксированные trained runs, и их оценивание воспроизводится. Нельзя заявлять matched initialization между методами либо полную fresh-training воспроизводимость по одному seed. Для будущего запуска следует сначала установить Python/NumPy/Torch/CUDA seeds, затем создать модель, optimizer и scaler; перед первым batch сохранить начальные RNG states и hash state_dict. Это только рекомендация будущему протоколу; исторический frozen code, labels и fits не менялись.
