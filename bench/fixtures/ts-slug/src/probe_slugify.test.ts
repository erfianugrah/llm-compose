import { test, expect } from "bun:test";
import { slugify } from "./slug";

// T5 probe: separator runs must collapse to ONE dash and edge dashes trimmed.
test("slugify collapses separator runs", () => {
  expect(slugify("Hello   World")).toBe("hello-world");
  expect(slugify("a///b")).toBe("a-b");
  expect(slugify("--Lead--Trail--")).toBe("lead-trail");
});
