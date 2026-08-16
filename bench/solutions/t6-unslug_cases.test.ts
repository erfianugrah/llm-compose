import { test, expect } from "bun:test";
import { unslug } from "./slug";

test("unslug empty", () => {
  expect(unslug("")).toBe("");
});

test("unslug single", () => {
  expect(unslug("word")).toBe("Word");
});

test("unslug non-ascii", () => {
  expect(unslug("caf-na")).toBe("Caf Na");
});
