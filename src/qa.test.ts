import { BertWordPieceTokenizer } from "tokenizers";
import { mocked } from "ts-jest";

import { QAClient } from "./qa";

describe("QAClient", () => {
  describe("fromOptions", () => {
    it("should instantiate a QAClient with custom tokenizer when provided", async () => {
      const tokenizer = jest.fn();
      const qaClient = await QAClient.fromOptions({
        tokenizer: (tokenizer as unknown) as BertWordPieceTokenizer
      });
      expect((qaClient as any).tokenizer).toBe(tokenizer);
    });
  });

  describe("predict", () => {
    let qa: QAClient;

    beforeEach(async () => {
      qa = await QAClient.fromOptions();
    });

    it("should call the model", async () => {
      const contexts = [
        `
          Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
          The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
          As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
        `,
        `
          One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745.
        `
      ];

      const queries = ["Who won the Super Bowl?", "Where was Chopin born?"];

      const predOne = await qa.predict(queries[0], contexts[0]);
      console.log(predOne);

      const predTwo = await qa.predict(queries[1], contexts[1]);
      console.log(predTwo);
    });
  });
});
