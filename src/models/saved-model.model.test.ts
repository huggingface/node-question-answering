import { SavedModel } from ".";

describe("SavedModel", () => {
  describe("fromOptions", () => {
    it("instantiates a QAClient with a partial path for the model", async () => {
      const model = await SavedModel.fromOptions({ path: "distilbert-cased" });
      expect(model).toBeDefined();
    });
  });
});
