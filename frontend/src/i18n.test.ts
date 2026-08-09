import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TEXT, initialLang, localeOf, rememberLang, type Text } from "./i18n";

const LANGS = ["ja", "en"] as const;
const keys = Object.keys(TEXT.ja) as (keyof Text)[];

describe("the dictionaries", () => {
  it("carry the same keys in both languages", () => {
    // Types enforce this at build time; this catches a key that was declared
    // and then left holding a copy of the other language's placeholder.
    expect(Object.keys(TEXT.en).sort()).toEqual(keys.slice().sort());
  });

  it("have no empty strings", () => {
    for (const lang of LANGS) {
      for (const key of keys) {
        const value = TEXT[lang][key];
        if (typeof value === "string") expect(value.trim()).not.toBe("");
      }
    }
  });

  it("agree on which entries take numbers", () => {
    for (const key of keys) {
      expect(typeof TEXT.en[key]).toBe(typeof TEXT.ja[key]);
    }
  });

  it("are actually translated, not copied", () => {
    // A handful of strings legitimately match across languages (units, the
    // OSM suffix); most must not, or a language was left half-done.
    const same = keys.filter((k) =>
      typeof TEXT.ja[k] === "string" && TEXT.ja[k] === TEXT.en[k]);
    expect(same.length).toBeLessThan(keys.length * 0.1);
  });

  it("keep the bold marker balanced in every hint", () => {
    // App.tsx's rich() renders the odd-indexed runs bold; an unpaired * would
    // silently bold the rest of the sentence.
    for (const lang of LANGS) {
      for (const key of keys) {
        const value = TEXT[lang][key];
        if (typeof value === "string") {
          expect(value.split("*").length % 2).toBe(1);
        }
      }
    }
  });
});

describe("duration", () => {
  it.each([
    [0, "0分", "0 min"],
    [90, "2分", "2 min"],
    [3600, "1時間00分", "1h 00m"],
    [3660 + 240, "1時間05分", "1h 05m"],
  ])("reads %s seconds the way each language does", (sec, ja, en) => {
    expect(TEXT.ja.duration(sec)).toBe(ja);
    expect(TEXT.en.duration(sec)).toBe(en);
  });
});

describe("plateFit", () => {
  it("names the area in both languages", () => {
    expect(TEXT.ja.plateFit(40, 16)).toContain("40×16mm");
    expect(TEXT.en.plateFit(40, 16)).toContain("40×16mm");
  });
});

describe("localeOf", () => {
  it("picks the locale dates are formatted with", () => {
    expect(localeOf("ja")).toBe("ja-JP");
    expect(localeOf("en")).toBe("en-US");
  });
});

describe("initialLang", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => vi.unstubAllGlobals());

  const withLanguage = (language: string) =>
    vi.stubGlobal("navigator", { ...navigator, language });

  it("remembers the last choice above all else", () => {
    withLanguage("en-US");
    rememberLang("ja");
    expect(initialLang()).toBe("ja");
  });

  it("gives a Japanese browser Japanese", () => {
    withLanguage("ja-JP");
    expect(initialLang()).toBe("ja");
  });

  it("gives everyone else English", () => {
    withLanguage("de-DE");
    expect(initialLang()).toBe("en");
  });

  it("ignores a stored value that is not a language", () => {
    withLanguage("ja");
    localStorage.setItem("3dfp.lang", "klingon");
    expect(initialLang()).toBe("ja");
  });

  it("survives a browser that reports no language at all", () => {
    vi.stubGlobal("navigator", {});
    expect(initialLang()).toBe("en");
  });
});
