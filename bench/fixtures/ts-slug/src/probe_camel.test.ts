import { test, expect } from "bun:test";
import { camelCase } from "./slug";

// T4 probe: camelCase("hello-world") -> "helloWorld".
test("camelCase basic", () => {
  expect(camelCase("hello-world")).toBe("helloWorld");
  expect(camelCase("a--b-c")).toBe("aBC");
  expect(camelCase("")).toBe("");
});
