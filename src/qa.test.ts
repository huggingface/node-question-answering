import { BertWordPieceTokenizer } from "tokenizers";
import { mocked } from "ts-jest";

import { TFJSModel } from "./models";
import { QAClient } from "./qa";

const basicQuestion = "Who won the Super Bowl?";
const basicContext = `
  Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
  The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
  As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
`;

describe("QAClient", () => {
  describe("fromOptions", () => {
    it("instantiates a QAClient with custom tokenizer when provided", async () => {
      const tokenizer = jest.fn();
      const qaClient = await QAClient.fromOptions({
        tokenizer: (tokenizer as unknown) as BertWordPieceTokenizer
      });
      expect((qaClient as any).tokenizer).toBe(tokenizer);
    });

    it("leads to answer without inference time by default", async () => {
      const qaClient = await QAClient.fromOptions();
      const predOne = await qaClient.predict(basicQuestion, basicContext);
      expect(predOne?.inferenceTime).toBeUndefined();
    });

    it("leads to answer with inference time when `timeIt` is `true`", async () => {
      const qaClient = await QAClient.fromOptions({ timeIt: true });
      const predOne = await qaClient.predict(basicQuestion, basicContext);
      expect(typeof predOne?.inferenceTime).toBe("number");
    });
  });

  describe("predict", () => {
    let qa: QAClient;

    const shorts = [
      {
        context: `
          Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season.
          The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California.
          As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.
        `,
        question: ["Who won the Super Bowl?", "Denver Broncos"]
      },
      {
        context: `
          One of the most famous people born in Warsaw was Maria Skłodowska-Curie, who achieved international recognition for her research on radioactivity and was the first female recipient of the Nobel Prize. Famous musicians include Władysław Szpilman and Frédéric Chopin. Though Chopin was born in the village of Żelazowa Wola, about 60 km (37 mi) from Warsaw, he moved to the city with his family when he was seven months old. Casimir Pulaski, a Polish general and hero of the American Revolutionary War, was born here in 1745.
        `,
        question: ["Where was Chopin born?", "Żelazowa Wola"]
      }
    ];

    const long = {
      context: `
        At his father's death on 16 September 1380, Charles VI inherited the throne of France. His coronation took place on 4 November 1380, at Reims Cathedral. Charles VI was only 11 years old when he was crowned King of France. During his minority, France was ruled by Charles' uncles, as regents. Although the royal age of majority was 14 (the "age of accountability" under Roman Catholic canon law), Charles terminated the regency only at the age of 21.

        The regents were Philip the Bold, Duke of Burgundy, Louis I, Duke of Anjou, and John, Duke of Berry – all brothers of Charles V – along with Louis II, Duke of Bourbon, Charles VI's maternal uncle. Philip took the dominant role during the regency. Louis of Anjou was fighting for his claim to the Kingdom of Naples after 1382, dying in 1384; John of Berry was interested mainly in the Languedoc, and not particularly interested in politics; and Louis of Bourbon was a largely unimportant figure, owing to his personality (showing signs of mental instability) and status (since he was not the son of a king).
        
        During the rule of his uncles, the financial resources of the kingdom, painstakingly built up by his father, were squandered for the personal profit of the dukes, whose interests were frequently divergent or even opposing. During that time, the power of the royal administration was strengthened and taxes re-established. The latter policy represented a reversal of the deathbed decision of the king's father Charles V to repeal taxes, and led to tax revolts, known as the Harelle. Increased tax revenues were needed to support the self-serving policies of the king's uncles, whose interests were frequently in conflict with those of the crown and with each other. The Battle of Roosebeke (1382), for example, brilliantly won by the royal troops, was prosecuted solely for the benefit of Philip of Burgundy. The treasury surplus carefully accumulated by Charles V was quickly squandered.
        
        Charles VI brought the regency to an end in 1388, taking up personal rule. He restored to power the highly competent advisors of Charles V, known as the Marmousets, who ushered in a new period of high esteem for the crown. Charles VI was widely referred to as Charles the Beloved by his subjects.

        He married Isabeau of Bavaria on 17 July 1385, when he was 17 and she 14 (and considered an adult at the time). Isabeau had 12 children, most of whom died young. Isabeau's first child, named Charles, was born in 1386, and was Dauphin of Viennois (heir apparent), but survived only 3 months. Her second child, Joan, was born on 14 June 1388, but died in 1390. Her third child, Isabella, was born in 1389. She was married to Richard II, King of England in 1396, at the age of 6, and became Queen of England. Richard died in 1400 and they had no children. Richard's successor, Henry IV, wanted Isabella then to marry his son, 14-year-old future king Henry V, but she refused. She was married again in 1406, this time to her cousin, Charles, Duke of Orléans, at the age of 17. She died in childbirth at the age of 19.

        Isabeau's fourth child, Joan, was born in 1391, and was married to John VI, Duke of Brittany in 1396, at an age of 5; they had children. Isabeau's fifth child born in 1392 was also named Charles, and was Dauphin. The young Charles was betrothed to Margaret of Burgundy in 1396, but died at the age of 9. Isabeau's sixth child, Mary, was born in 1393. She was never married, and had no children. Isabeau's seventh child, Michelle, was born in 1395. She was engaged to Philip, son of John the Fearless, Duke of Burgundy, in 1404 (both were then aged 8) and they were married in 1409, aged 14. She had one child who died in infancy, before she died in 1422, aged 27.

        Isabeau's eighth child, Louis, was born in 1397, and was also Dauphin. He married Margaret of Burgundy, who had previously been betrothed to his brother Charles. The marriage produced no children by the time of Louis's death in 1415, aged 18.

        Isabeau's ninth child, John, was born in 1398, and was also Dauphin from 1415, after the death of his brother Louis. He was married to Jacqueline, Countess of Hainaut in 1415, then aged 17, but they did not have any children before he died in 1417, aged 19. Isabeau's tenth child, Catherine, was born in 1401. She was married firstly to Henry V, King of England in 1420, and they had one child, who became Henry VI of England. Henry V died suddenly in 1422. Catherine may then have secretly married Owen Tudor in 1429 and had additional children, including Edmund Tudor, the father of Henry VII. She died in 1437, aged 36.
      `,
      questions: [
        ["When did his father die?", "16 September 1380"],
        ["Who did Charles VI marry?", "Isabeau of Bavaria"],
        ["What was the name of Isabeau's eighth child?", "Louis"]
      ]
    };

    describe("using SavedModel format", () => {
      beforeEach(async () => {
        qa = await QAClient.fromOptions();
      });

      it.each(shorts)("gives the correct answer with short contexts", async short => {
        const predOne = await qa.predict(short.question[0], short.context);
        expect(predOne?.text).toEqual(short.question[1]);
      });

      for (const question of long.questions) {
        it("gives the correct answer with long contexts", async () => {
          const predOne = await qa.predict(question[0], long.context);
          expect(predOne?.text).toEqual(question[1]);
        });
      }
    });

    describe("using TFJS format", () => {
      beforeEach(async () => {
        const model = await TFJSModel.fromOptions({
          path: "distilbert-cased",
          cased: true
        });
        qa = await QAClient.fromOptions({ model });
      });

      it.each(shorts)("gives the correct answer with short contexts", async short => {
        const predOne = await qa.predict(short.question[0], short.context);
        expect(predOne?.text).toEqual(short.question[1]);
      });

      for (const question of long.questions) {
        it("gives the correct answer with long contexts", async () => {
          const predOne = await qa.predict(question[0], long.context);
          expect(predOne?.text).toEqual(question[1]);
        });
      }
    });
  });
});
