export interface StringStorage {
  getItem(key: string): string | null
  setItem(key: string, value: string): void
  removeItem(key: string): void
}

export function stored(
  storage: Pick<StringStorage, "getItem">,
  key: string,
  fallback: string,
  legacyKey?: string,
) {
  try { return storage.getItem(key) || (legacyKey ? storage.getItem(legacyKey) : null) || fallback }
  catch { return fallback }
}

export function persistPublicSettings(storage: StringStorage, baseUrl: string, model: string) {
  try {
    storage.setItem("shiftwing.baseUrl", baseUrl)
    storage.setItem("shiftwing.model", model)
    // API credentials intentionally remain memory-only. Remove values left by
    // older web releases whenever public settings are persisted.
    storage.removeItem("shiftwing.apiKey")
    storage.removeItem("colibri.apiKey")
    storage.removeItem("colibri.baseUrl")
    storage.removeItem("colibri.model")
  } catch { /* restricted storage mode */ }
}
